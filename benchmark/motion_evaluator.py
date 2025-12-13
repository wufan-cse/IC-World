import os
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from utils import video_to_image_sequence_tensor, split_grid_video

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
THIRDPARTY_DIR = os.path.join(ROOT_DIR, "thirdparty")
SPATRACKER_DIR = os.path.join(ROOT_DIR, "thirdparty", "SpaTrackerV2")

sys.path.insert(0, THIRDPARTY_DIR)
sys.path.insert(0, SPATRACKER_DIR)

from SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor as Predictor
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track as VGGT4Track
from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from SpaTrackerV2.models.SpaTrackV2.models.utils import get_points_on_a_grid


class MotionEvaluator:
    def __init__(
        self,
        device: torch.device,
        interval: int = 10,
        grid_size: int = 10,
        vo_points: int = 756,
        tracker_model_path: str = None,
        vggt_model_path: str = None
    ):
        self.device = device
        self.interval = interval
        self.grid_size = grid_size
        self.vo_points = vo_points        

        # Load models
        self.tracker_model = Predictor.from_pretrained(
            pretrained_model_name_or_path=tracker_model_path
        )
        self.tracker_model.to(self.device) 
        self.tracker_model.eval()
        self.tracker_model.spatrack.track_num = int(self.vo_points)

        self.vggt_model = VGGT4Track.from_pretrained(
            pretrained_model_name_or_path=vggt_model_path
        )
        self.vggt_model.to(self.device)
        self.vggt_model.eval()

    def _prepare_inputs(self, seq: torch.Tensor):

        with torch.no_grad():
            vid = (seq * 255.0).clamp(0, 255).to(torch.uint8)  # (N,3,H,W) uint8
            vid = vid.float()
            video_tensor = preprocess_image(vid)[None].to(self.device)


            predictions = self.vggt_model((video_tensor / 255.0))
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

        depth_tensor = depth_map.squeeze()
        extrs = extrinsic.squeeze()
        intrs = intrinsic.squeeze()
        video_tensor = video_tensor.squeeze(0)  # (N,3,h,w) float32 tensor
        unc_metric = (depth_conf.squeeze() > 0.5)

        return video_tensor, depth_tensor, intrs, extrs, unc_metric

    def _run_spatracker_pipeline(self, seq: torch.Tensor, debug_name: str = None):
        # Returns tuple: (track3d_pred: (T, P, 3), c2w_traj: (T, 4, 4))
        # Models should be loaded and on correct device at this point
        # (handled by from_sequences method)
        video_tensor, depth_tensor, intrs, extrs, unc_metric = self._prepare_inputs(seq)

        # Build query grid points in image coordinates (HxW of processed frames)
        frame_H, frame_W = int(video_tensor.shape[1]), int(video_tensor.shape[2])  # numpy array shape is (N,H,W,C)
        grid_pts = get_points_on_a_grid(self.grid_size, (frame_H, frame_W))
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].cpu().numpy()

        with torch.no_grad():
            # Ensure tracker model is on the correct device            
            (
                c2w_traj,
                intrs_out,
                point_map,
                conf_depth,
                track3d_pred,
                track2d_pred,
                vis_pred,
                conf_pred,
                video_out,
            ) = self.tracker_model.forward(
                video_tensor,  # numpy array
                depth=depth_tensor,
                intrs=intrs,
                extrs=extrs,
                queries=query_xyt,
                fps=1,
                full_point=True,
                iters_track=4,
                query_no_BA=True,
                fixed_cam=False,
                stage=1,
                unc_metric=unc_metric,
                support_frame=max(1, len(video_tensor) - 1),
                replace_ratio=0.2,
            )

        if track3d_pred is None:
            return None, None

        # Ensure all tensors are on the same device and dtype
        c2w_traj = c2w_traj.to(self.device)
        track3d_pred = track3d_pred.to(self.device)
        
        R = c2w_traj[:, :3, :3]  # (T, 3, 3)
        t = c2w_traj[:, :3, 3]   # (T, 3)
        
        # Handle both tensor and numpy array inputs
        pts_cam = track3d_pred[:, :, :3]  # (T,P,3)
        
        # world = R @ cam + t
        pts_world = torch.einsum("tij,tpj->tpi", R, pts_cam) + t[:, None, :]
        tracks = pts_world
        
        return tracks.cpu().numpy(), c2w_traj.cpu().numpy() # (T, P, 3), (T, 4, 4)
            
    def _motion_consistency_distance(
        self, 
        tracks_left: np.ndarray, 
        tracks_right: np.ndarray,                              
        c2w_left: np.ndarray, 
        c2w_right: np.ndarray
    ) -> torch.Tensor:
        # tracks_*: (T, P, 3) numpy - tracks in world coordinates
        # c2w_*: (T, 4, 4) numpy - camera-to-world transformation matrices
        
        if tracks_left.ndim != 3 or tracks_right.ndim != 3:
            return torch.tensor(1e3, device=self.device, dtype=torch.float32)
        T_l, P_l, _ = tracks_left.shape
        T_r, P_r, _ = tracks_right.shape
        if T_l < 2 or T_r < 2 or P_l == 0 or P_r == 0:
            return torch.tensor(1e3, device=self.device, dtype=torch.float32)

        # Use camera poses to align the two sequences
        # The key insight: both sequences have predicted camera trajectories
        # We align the left sequence to the right sequence's coordinate system
        # using the relationship between their camera poses
        
        # Use the first camera pose from each sequence to establish alignment
        # c2w: camera to world, so w2c = inv(c2w)
        c2w_left_0 = c2w_left[0]  # (4, 4) - first frame's camera-to-world
        c2w_right_0 = c2w_right[0]  # (4, 4)
        
        # Compute world-to-camera for first frame of each sequence
        w2c_left_0 = np.linalg.inv(c2w_left_0)  # (4, 4)
        w2c_right_0 = np.linalg.inv(c2w_right_0)  # (4, 4)
        
        # Transform left tracks from left's world to left's camera frame 0, 
        # then to right's camera frame 0, then to right's world
        align_transform = c2w_right_0 @ w2c_left_0  # (4, 4)
        
        # Apply transformation to left tracks
        R_align = align_transform[:3, :3]  # (3, 3)
        t_align = align_transform[:3, 3]  # (3,)
        
        # tracks_left_aligned: (T, P_l, 3)
        tracks_left_aligned = (tracks_left @ R_align.T) + t_align[None, None, :]

        # Use time overlap
        min_time = min(T_l, T_r)
        if min_time == 0:
            return torch.tensor(1e3, device=self.device, dtype=torch.float32)
        
        tracks_left_aligned = tracks_left_aligned[:min_time]  # (T, P_l, 3)
        tracks_right = tracks_right[:min_time]  # (T, P_r, 3)
        
        # Compute per-point average positions for matching
        avg_pos_l = tracks_left_aligned.mean(axis=0)  # (P_l, 3)
        avg_pos_r = tracks_right.mean(axis=0)  # (P_r, 3)
        
        # Match each left point to nearest right point by average position
        # Then compute position distance between matched pairs across all frames
        motion_dists = []
        
        for i in range(P_l):
            pos_l = avg_pos_l[i]
            dists = np.linalg.norm(avg_pos_r - pos_l[None, :], axis=1)
            j = int(np.argmin(dists))
            
            # Compute position distance between matched tracks across all frames
            # tracks_left_aligned[:, i, :] : (T, 3) - positions of left point i over time
            # tracks_right[:, j, :] : (T, 3) - positions of right point j over time
            pos_diff = tracks_left_aligned[:, i, :] - tracks_right[:, j, :]  # (T, 3)
            pos_dist = np.linalg.norm(pos_diff, axis=1)  # (T,) - euclidean distance at each timestep
            mean_pos_dist = pos_dist.mean()  # Average distance across time
            
            motion_dists.append(mean_pos_dist)
        
        if len(motion_dists) == 0:
            return torch.tensor(1e3, device=self.device, dtype=torch.float32)
        
        # Return mean position distance
        return torch.tensor(np.mean(motion_dists), device=self.device, dtype=torch.float32) 

    @torch.no_grad()
    def from_sequences(self, left_seq: torch.Tensor, right_seq: torch.Tensor) -> torch.Tensor:
        """
        left_seq/right_seq: (N,3,H,W) in [0,1], torch.float32 on any device
        Returns: scalar reward tensor on self.device
        """
        tracks_left, c2w_left = self._run_spatracker_pipeline(left_seq)
        tracks_right, c2w_right = self._run_spatracker_pipeline(right_seq)

        if tracks_left is None or tracks_right is None or c2w_left is None or c2w_right is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)

        motion_distance = self._motion_consistency_distance(tracks_left, tracks_right, c2w_left, c2w_right)
        motion_consistency_score = torch.exp(-motion_distance)
        motion_consistency_score = torch.clamp(motion_consistency_score, 0.0, 1.0)
        return motion_consistency_score

    @torch.no_grad()
    def from_video_path(self, path: str) -> torch.Tensor:
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
            # Return tensor on CPU to avoid device issues
            return torch.tensor(0.0, device='cpu', dtype=torch.float32)

        video_np = np.stack(frames, axis=0)  # (T,H,W,3), uint8
        left_np, right_np = split_grid_video(video_np)
        left_seq = video_to_image_sequence_tensor(left_np, self.interval)
        right_seq = video_to_image_sequence_tensor(right_np, self.interval)
        # ensure minimal length
        min_len = max(2, min(left_seq.shape[0], right_seq.shape[0]))
        left_seq = left_seq[:min_len]
        right_seq = right_seq[:min_len]
        if left_seq.numel() == 0 or right_seq.numel() == 0:
            return torch.tensor(0.0, device='cpu', dtype=torch.float32)
        return self.from_sequences(left_seq, right_seq)

def evaluate(
    video_dir: str, 
    interval: int = 10,
    grid_size: int = 10,
    tracker_model_path: str = None,
    vggt_model_path: str = None,
    ) -> Dict[str, any]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print(f"Evaluating videos in: {video_dir}")
    print("=" * 80)
    
    print(f"\nInitializing Motion Evaluator...")
    print(f"  - Device: {device}")
    print(f"  - Grid size: {grid_size}")
    print(f"  - Interval: {interval}")
    
    evaluator = MotionEvaluator(
        device=device,
        grid_size=grid_size,
        interval=interval,
        tracker_model_path=tracker_model_path,
        vggt_model_path=vggt_model_path
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
        score = evaluator.from_video_path(str(video_path))
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
    print(f"  Grid size:  {grid_size}")
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
        description="Motion Consistency Evaluator"
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        help='Directory containing videos to evaluate (required for evaluate mode)'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=10,
        help='Grid size for query points'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Frame sample interval'
    )
    parser.add_argument(
        '--tracker_model_path',
        type=str,
        default="weights/SpatialTrackerV2-Offline",
        help='Path to tracker model (default: weights/SpatialTrackerV2-Offline)'
    )
    parser.add_argument(
        '--vggt_model_path',
        type=str,
        default="weights/SpatialTrackerV2_Front",
        help='Path to VGGT model (default: weights/SpatialTrackerV2_Front)'
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
            grid_size=args.grid_size,
            tracker_model_path=args.tracker_model_path,
            vggt_model_path=args.vggt_model_path
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
