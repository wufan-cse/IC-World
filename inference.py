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
import numpy as np


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


model_id = "./weights/IC-World-I2V-14B"
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)

pipe.load_lora_weights("./weights/IC-World-I2V-14B", weight_name="pytorch_lora_weights.safetensors", prefix="transformer")
pipe.to("cuda")


input_image1 = Image.open("./assets/img.png").convert("RGB")
input_image2 = Image.open("./assets/img1.png").convert("RGB")

input_image1 = crop_and_resize(input_image1, 480, 416)
input_image2 = crop_and_resize(input_image2, 480, 416)

image = Image.fromarray(
    np.hstack([np.array(input_image1), np.array(input_image2)])
)

prompt = ""
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

generator = torch.Generator(device="cuda").manual_seed(42)

output = pipe(
    image=image, prompt=prompt, negative_prompt=negative_prompt, height=480, width=832, num_frames=49, guidance_scale=1.0, num_inference_steps=4, generator=generator
).frames[0]

export_to_video(output, "output.mp4", fps=16)
