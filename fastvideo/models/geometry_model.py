import torch
import numpy as np
import sys
import os
import argparse
import cv2
import yaml

from PIL import Image
from easydict import EasyDict as edict
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from fastvideo.models.utils import video_to_image_sequence_tensor, split_grid_video

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
LEPARD_DIR = os.path.join(ROOT_DIR, "thirdparty", "lepard")

sys.path.insert(0, THIRDPARTY_DIR)
sys.path.insert(0, LEPARD_DIR)

from Pi3.pi3.models.pi3 import Pi3 as Pi3Model
from lepard.inference import Pipeline
from lepard.configs.models import architectures


class GeometryModel:
    def __init__(
        self, 
        device: torch.device, 
        interval: int = 10,
        confidence_threshold: float = 0.1,
        pi3_model_path: str = None,
        lepard_config_path: str = None,
        lepard_model_path: str = None
    ):
        self.device = device
        self.interval = interval
        self.confidence_threshold = confidence_threshold

        # Load Pi3
        self.pi3 = Pi3Model.from_pretrained(pretrained_model_name_or_path=pi3_model_path).to(device).eval()
        
        # Load LEPARD model
        def join(loader, node):
            seq = loader.construct_sequence(node)
            return '_'.join([str(i) for i in seq])
            
        yaml.add_constructor('!join', join)
        with open(lepard_config_path, 'r') as f:
            lepard_config = yaml.load(f, Loader=yaml.Loader)
        lepard_config = edict(lepard_config)
        lepard_config.kpfcn_config.architecture = architectures.get(lepard_config.get('dataset', '3dmatch'), architectures['3dmatch'])
        
        # Initialize LEPARD model
        self.lepard_model = Pipeline(lepard_config)
        model_state_dict = torch.load(lepard_model_path, map_location="cpu")["state_dict"]
        self.lepard_model.load_state_dict(model_state_dict)
        self.lepard_model.eval()
    
    @staticmethod
    def _procrustes_align_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Procrustes alignment distance between two point sets.
        Higher values indicate more different trajectories.
        
        Args:
            x: First point set (N, 3)
            y: Second point set (M, 3)
            
        Returns:
            Distance metric (higher = more different)
        """
        if x.shape[0] != y.shape[0]:
            n = min(x.shape[0], y.shape[0])
            x = x[:n]
            y = y[:n]
        x_mean = x.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        x0 = x - x_mean
        y0 = y - y_mean
        x_norm = torch.norm(x0)
        y_norm = torch.norm(y0)
        if x_norm.item() == 0 or y_norm.item() == 0:
            return torch.tensor(1e3, device=x.device, dtype=x.dtype)
        x0 = x0 / x_norm
        y0 = y0 / y_norm
        H = x0.T @ y0
        U, _, Vt = torch.linalg.svd(H)
        R = U @ Vt
        if torch.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        x_aligned = (x0 @ R) * (y_norm / x_norm)
        mse = torch.mean((x_aligned - y0) ** 2)
        return mse  

    @staticmethod
    def _downsample_points(pts: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
        if pts.shape[0] <= max_points:
            return pts
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        return pts[idx]

    def _mask_points(self, output: dict) -> torch.Tensor:
        """
        Build reliability mask using confidence and non-edge depth heuristic.
        
        Args:
            output: Pi3 output dict with keys: points (1,N,H,W,3), conf (1,N,H,W,1)
            
        Returns:
            Masked point cloud (M, 3)
        """
        # output keys: points (1,N,H,W,3), conf (1,N,H,W,1)
        points = output['points'][0]          # (N,H,W,3)
        conf = output['conf'][0][..., 0]      # (N,H,W)
        # Confidence threshold - configurable via self.confidence_threshold
        conf_mask = torch.sigmoid(conf) > self.confidence_threshold
        try:
            from Pi3.pi3.utils.geometry import depth_edge
            # Use local z for edge detection
            depth = output['local_points'][0][..., 2]
            non_edge = ~depth_edge(depth, rtol=0.03)
            mask = torch.logical_and(conf_mask, non_edge)
        except Exception:
            mask = conf_mask
        pts = points[mask]
        return pts  # (M,3)

    @staticmethod
    def _chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (M,3), b: (K,3)
        if a.numel() == 0 or b.numel() == 0:
            return torch.tensor(1e3, device=a.device if a.is_cuda else b.device)
        dists = torch.cdist(a[None], b[None])[0]  # (M,K)
        a2b = dists.min(dim=1).values.mean()
        b2a = dists.min(dim=0).values.mean()
        return (a2b + b2a) * 0.5  

    @torch.no_grad()
    def _lepard_register(self, src_pts: torch.Tensor, tgt_pts: torch.Tensor) -> tuple:
        """
        Use LEPARD to register source to target point cloud.
        
        Args:
            src_pts: Source point cloud (N, 3)
            tgt_pts: Target point cloud (M, 3)
            
        Returns:
            R: Rotation matrix (3, 3)
            t: Translation vector (3, 1)
            registration_quality: Quality score of the registration
        """
        src_np = src_pts.cpu().numpy()
        tgt_np = tgt_pts.cpu().numpy()
        
        # Prepare data for LEPARD
        data = {
            'src': src_np,
            'tgt': tgt_np
        }
        
        # Run LEPARD registration
        result = self.lepard_model(data)
        
        # Extract transformation
        R = result['R_s2t_pred'][0].to(self.device)  # (3, 3)
        t = result['t_s2t_pred'][0].to(self.device)  # (3, 1)
        
        # Compute registration quality based on inlier matches
        if 'conf_matrix_pred' in result:
            conf_matrix = result['conf_matrix_pred'][0]
            quality = torch.max(conf_matrix).item()
        else:
            quality = 1.0
        
        return R, t, torch.tensor(quality, device=self.device)

    def _scene_distance(self, out_left: dict, out_right: dict) -> torch.Tensor:
        """
        Compute scene distance between two reconstructions.
        
        Args:
            out_left: Pi3 output for left video
            out_right: Pi3 output for right video
            
        Returns:
            Distance metric (lower = more similar geometry)
        """
        pts_l = self._mask_points(out_left).to(self.device)
        pts_r = self._mask_points(out_right).to(self.device)
        pts_l = self._downsample_points(pts_l)
        pts_r = self._downsample_points(pts_r)
        
        if pts_l.numel() == 0 or pts_r.numel() == 0:
            return torch.tensor(1e3, device=self.device)
        
        # Ensure we have enough points
        min_points = min(pts_l.shape[0], pts_r.shape[0])
        if min_points < 3: 
            return torch.tensor(1e3, device=self.device)
        
        R, t, quality = self._lepard_register(pts_l, pts_r)
        
        pts_l_aligned = pts_l @ R.T + t.T
        
        distance = self._chamfer_distance(pts_l_aligned, pts_r)
        
        return distance
    
    def _camera_trajectory_diversity(self, out_left: dict, out_right: dict) -> torch.Tensor:
        """
        Compute camera trajectory diversity between two sequences.
        Higher values indicate more different camera trajectories.
        
        Args:
            out_left: Pi3 output for left video
            out_right: Pi3 output for right video
            
        Returns:
            Diversity metric (higher = more different trajectories)
        """
        poses_left = out_left['camera_poses'][0]
        poses_right = out_right['camera_poses'][0]
        centers_left = poses_left[:, :3, 3]  # Extract camera positions
        centers_right = poses_right[:, :3, 3]
        
        # Compute Procrustes alignment distance
        diversity = self._procrustes_align_distance(centers_left, centers_right)
        
        return diversity

    @torch.no_grad()
    def from_sequences(
        self, 
        left_seq: torch.Tensor, 
        right_seq: torch.Tensor,
        mode: str = "scene"
    ) -> torch.Tensor:
        """
        Evaluate sequences based on different modes.
        
        Args:
            left_seq/right_seq: (N,3,H,W) in [0,1], torch.float32 on any device
            mode: Evaluation mode, one of:
                - "scene": Geometry consistency (default, 0=different, 1=similar)
                - "camera": Camera trajectory diversity (0=similar, 1=different)
                - "combined": Combined reward balancing geometry consistency and camera diversity
                              (0=bad, 1=good: high geometry consistency + high camera diversity)
        
        Returns:
            Scalar score tensor on self.device
        """
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        left_seq = left_seq.to(self.device)
        right_seq = right_seq.to(self.device)

        with torch.amp.autocast('cuda', dtype=dtype):
            out_left = self.pi3(left_seq[None])
            out_right = self.pi3(right_seq[None])

        if mode == "scene":
            # Geometry consistency evaluation (default)
            scene_dist = self._scene_distance(out_left, out_right)
            # Normalize scene distance to [0, 1] range (0 = different, 1 = similar)
            scene_norm = torch.exp(-scene_dist)
            scene_norm = torch.clamp(scene_norm, 0.0, 1.0)
            return scene_norm
        
        elif mode == "camera":
            # Camera trajectory diversity evaluation
            camera_diversity = self._camera_trajectory_diversity(out_left, out_right)
            # Normalize diversity to [0, 1] range (0 = similar, 1 = different)
            # Use sigmoid to map diversity to [0, 1]
            camera_norm = torch.sigmoid(camera_diversity)
            camera_norm = torch.clamp(camera_norm, 0.0, 1.0)
            return camera_norm
        
        elif mode == "combined":
            # Combined reward: encourage different camera trajectories while maintaining geometry consistency
            scene_dist = self._scene_distance(out_left, out_right)
            camera_diversity = self._camera_trajectory_diversity(out_left, out_right)
            
            # Normalize both metrics to [0, 1] range
            scene_norm = torch.exp(-scene_dist)  # 0=different, 1=similar
            camera_norm = torch.sigmoid(camera_diversity)  # 0=similar, 1=different
            
            # Ensure values are in valid range
            scene_norm = torch.clamp(scene_norm, 0.0, 1.0)
            camera_norm = torch.clamp(camera_norm, 0.0, 1.0)
            
            # Combine: high geometry consistency (scene_norm) + high camera diversity (camera_norm)
            combined = 0.8 * scene_norm + 0.2 * camera_norm # TODO: preset weights, check out latest technique, GDPO: https://arxiv.org/abs/2601.05242
            
            # Final safety check
            if not torch.isfinite(combined):
                combined = torch.tensor(0.0, device=self.device, dtype=scene_norm.dtype)
            
            return torch.clamp(combined, 0.0, 1.0)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of: 'scene', 'camera', 'combined'")

    @torch.no_grad()
    def from_video_path(self, path: str, mode: str = "scene") -> torch.Tensor:
        """
        Evaluate a grid video (side-by-side left|right).
        
        Args:
            path: Path to grid video file
            mode: Evaluation mode ("scene", "camera", or "combined")
        
        Returns:
            Scalar score tensor
        """
        cap = cv2.VideoCapture(os.path.abspath(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        if len(frames) < 2:
            return torch.tensor(-1.0, device=self.device)
        video_np = np.stack(frames, axis=0)
        left_np, right_np = split_grid_video(video_np)
        left_seq = video_to_image_sequence_tensor(left_np, self.interval)
        right_seq = video_to_image_sequence_tensor(right_np, self.interval)
        # ensure minimal length
        min_len = max(2, min(left_seq.shape[0], right_seq.shape[0]))
        left_seq = left_seq[:min_len]
        right_seq = right_seq[:min_len]
        if left_seq.numel() == 0 or right_seq.numel() == 0:
            return torch.tensor(-1.0, device=self.device)
        return self.from_sequences(left_seq, right_seq, mode=mode)
    
    @torch.no_grad()
    def get_geometry_consistency_score(self, left_seq: torch.Tensor, right_seq: torch.Tensor) -> torch.Tensor:
        """
        Calculate geometry consistency score between two sequences.
        Lower values indicate better geometry consistency (raw distance).
        
        Args:
            left_seq/right_seq: (N,3,H,W) in [0,1], torch.float32
        
        Returns:
            Raw scene distance (lower = more consistent)
        """
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        left_seq = left_seq.to(self.device)
        right_seq = right_seq.to(self.device)
        
        with torch.amp.autocast('cuda', dtype=dtype):
            out_left = self.pi3(left_seq[None])
            out_right = self.pi3(right_seq[None])
        
        return self._scene_distance(out_left, out_right)
    
    @torch.no_grad()
    def get_camera_trajectory_diversity_score(self, left_seq: torch.Tensor, right_seq: torch.Tensor) -> torch.Tensor:
        """
        Calculate camera trajectory diversity score between two sequences.
        Higher values indicate more different trajectories (raw diversity).
        
        Args:
            left_seq/right_seq: (N,3,H,W) in [0,1], torch.float32
        
        Returns:
            Raw camera trajectory diversity (higher = more different)
        """
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        left_seq = left_seq.to(self.device)
        right_seq = right_seq.to(self.device)
        
        with torch.amp.autocast('cuda', dtype=dtype):
            out_left = self.pi3(left_seq[None])
            out_right = self.pi3(right_seq[None])
        
        return self._camera_trajectory_diversity(out_left, out_right)


def evaluate(
    video_dir: str, 
    confidence_threshold: float = 0.1, 
    interval: int = 10,
    pi3_model_path: str = None,
    lepard_config_path: str = None,
    lepard_model_path: str = None,
    mode: str = "scene"
    ) -> Dict[str, any]:
    """
    Evaluate videos in a directory.
    
    Args:
        video_dir: Directory containing videos to evaluate
        confidence_threshold: Confidence threshold for point filtering
        interval: Frame sample interval
        pi3_model_path: Path to Pi3 model
        lepard_config_path: Path to LEPARD config
        lepard_model_path: Path to LEPARD model
        mode: Evaluation mode ("scene", "camera", or "combined")
    
    Returns:
        Dictionary containing evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print(f"Evaluating videos in: {video_dir}")
    print("=" * 80)
    
    print(f"\nInitializing Geometry Evaluator...")
    print(f"  - Device: {device}")
    print(f"  - Confidence threshold: {confidence_threshold}")
    print(f"  - Interval: {interval}")
    print(f"  - Mode: {mode}")
    
    evaluator = GeometryModel(
        device=device,
        confidence_threshold=confidence_threshold,
        interval=interval,
        pi3_model_path=pi3_model_path,
        lepard_config_path=lepard_config_path,
        lepard_model_path=lepard_model_path
    )
    
    # Find all video files
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir_path.glob(f"*{ext}")))
        video_files.extend(list(video_dir_path.glob(f"*{ext.upper()}")))
    
    video_files = sorted(list(set(video_files)))  # Remove duplicates and sort
    
    if len(video_files) == 0:
        raise ValueError(f"No video files found in: {video_dir}")
    
    print(f"\nFound {len(video_files)} video file(s):")
    
    # Evaluate each video
    print("\n" + "-" * 80)
    print("Evaluating videos...")
    print("-" * 80)
    
    scores = []
    failed_videos = []
    
    for i, video_path in enumerate(tqdm(video_files, desc="Processing videos"), 1):
        score = evaluator.from_video_path(str(video_path), mode=mode)
        scores.append(score.item() if torch.is_tensor(score) else score)        
    
    # Calculate statistics
    if len(scores) == 0:
        raise RuntimeError("All videos failed to process")
    
    scores_array = np.array(scores)
    mean_score = np.mean(scores_array)
    std_score = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    print(f"\nSuccessfully evaluated: {len(scores)}/{len(video_files)} videos")
    
    if failed_videos:
        print(f"\nFailed videos ({len(failed_videos)}):")
        for video_path, error in failed_videos:
            print(f"  - {Path(video_path).name}: {error}")
    
    print("\nScore Statistics:")
    print(f"  Confidence_threshold:  {confidence_threshold:.6f}")
    print(f"  Mean:  {mean_score:.6f}")
    print(f"  Std:   {std_score:.6f}")
    print(f"  Min:   {min_score:.6f}")
    print(f"  Max:   {max_score:.6f}")
    print(f"  Count: {len(scores)}")
    
    print("\n" + "=" * 80)
    
    # Return results
    results = {
        'scores': scores,
        'video_files': [str(vf) for vf in video_files[:len(scores)]],
        'failed_videos': failed_videos,
        'mean': mean_score,
        'std': std_score,
        'min': min_score,
        'max': max_score,
        'count': len(scores)
    }
    
    return results


def main():
    """Run all tests or evaluate videos based on command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Geometry Consistency Evaluator"
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        help='Directory containing videos to evaluate (required for evaluate mode)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for point filtering (default: 0.1)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Frame sample interval'
    )
    parser.add_argument(
        '--pi3_model_path',
        type=str,
        default="weights/Pi3",
    )
    parser.add_argument(
        '--lepard_config_path',
        type=str,
        default="thirdparty/lepard/configs/test/3dmatch.yaml",
    )
    parser.add_argument(
        '--lepard_model_path',
        type=str,
        default="weights/lepard/pretrained/3dmatch/model_best_loss.pth",
    )
    parser.add_argument(
        '--mode',
        type=str,
        default="scene",
        choices=["scene", "camera", "combined"],
        help='Evaluation mode: "scene" (geometry consistency), "camera" (trajectory diversity), or "combined" (both)'
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    
    if not args.video_dir:
        print("Error: --video_dir is required for evaluate mode")
        parser.print_help()
        return False
    
    try:
        results = evaluate(
            video_dir=args.video_dir,
            interval=args.interval,
            confidence_threshold=args.confidence_threshold,
            pi3_model_path=args.pi3_model_path,
            lepard_config_path=args.lepard_config_path,
            lepard_model_path=args.lepard_model_path,
            mode=args.mode
        )
        return True
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
