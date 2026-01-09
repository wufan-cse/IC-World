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


def video_to_image_sequence_tensor(video_np: np.ndarray, interval: int = 10) -> torch.Tensor:
    frames = []
    T = video_np.shape[0]
    for i in range(0, T, interval):
        frame = video_np[i]
        frame = torch.from_numpy(frame).to(torch.float32) / 255.0
        frame = frame.permute(2, 0, 1)  # (C, H, W)
        # Resize to ensure dimensions are multiples of 14 (Pi3 patch size)
        C, H, W = frame.shape
        new_H = ((H + 13) // 14) * 14  # Round up to nearest multiple of 14
        new_W = ((W + 13) // 14) * 14
        if H != new_H or W != new_W:
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), 
                size=(new_H, new_W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        frames.append(frame)
    if not frames:
        return torch.empty(0)
    return torch.stack(frames, dim=0)

def split_grid_video(video_np: np.ndarray):
    T, H, W, C = video_np.shape
    mid = W // 2
    left_np = video_np[:, :, :mid, :]
    right_np = video_np[:, :, mid:, :]
    return left_np, right_np