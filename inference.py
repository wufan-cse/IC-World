import argparse
import torch
import torch.nn as nn
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

import torchvision

from diffusers import AutoencoderKLWan, WanTransformer3DModel
from PIL import Image

from safetensors.torch import load_file
from safetensors import safe_open

import re


def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image


def split_grid_video(video_np: np.ndarray):
    """
    Split a NumPy array of frames (T, H, W, C) into left and right halves.
    Returns two lists of PIL.Image.
    """
    T, H, W, C = video_np.shape
    mid = W // 2

    left_frames = video_np[:, :, :mid, :]
    right_frames = video_np[:, :, mid:, :]

    return left_frames, right_frames


def parse_args():
    parser = argparse.ArgumentParser(description="IC-World I2V Inference Script")
    
    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./weights/IC-World-I2V-14B",
                        help="Path to the model directory")
    parser.add_argument("--lora_weights_path", type=str, default="./weights/IC-World-I2V-14B",
                        help="Path to LoRA weights directory")
    parser.add_argument("--lora_weight_name", type=str, default="pytorch_lora_weights.safetensors",
                        help="Name of the LoRA weights file")
    
    # Input image arguments
    parser.add_argument("--input_image1", type=str, default="./assets/img.png",
                        help="Path to the first input image")
    parser.add_argument("--input_image2", type=str, default="./assets/img1.png",
                        help="Path to the second input image")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="Negative prompt for generation")
    
    # Video parameters
    parser.add_argument("--height", type=int, default=480,
                        help="Output video height")
    parser.add_argument("--width", type=int, default=832,
                        help="Output video width")
    parser.add_argument("--num_frames", type=int, default=49,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second for output video")
    
    # Inference parameters
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=4,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for generation")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output video file path")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.pretrained_model_name_or_path}...")
    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(args.pretrained_model_name_or_path, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
    
    # Load LoRA weights
    print(f"Loading LoRA weights from {args.lora_weights_path}...")
    pipe.load_lora_weights(args.lora_weights_path, weight_name=args.lora_weight_name, prefix="transformer")
    pipe.to(args.device)
    
    # Load and process input images
    print(f"Loading input images...")
    input_image1 = Image.open(args.input_image1).convert("RGB")
    input_image2 = Image.open(args.input_image2).convert("RGB")
    
    # Calculate individual image dimensions (half of total width)
    individual_width = args.width // 2
    individual_height = args.height
    
    input_image1 = crop_and_resize(input_image1, individual_height, individual_width)
    input_image2 = crop_and_resize(input_image2, individual_height, individual_width)
    
    image = Image.fromarray(
        np.hstack([np.array(input_image1), np.array(input_image2)])
    )
    
    # Set up generator
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Generate video
    print(f"Generating video with {args.num_frames} frames...")
    output = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator
    ).frames[0]
    
    # Export video
    print(f"Saving video to {args.output}...")
    export_to_video(output, args.output, fps=args.fps)
    print("Done!")


if __name__ == "__main__":
    main()
