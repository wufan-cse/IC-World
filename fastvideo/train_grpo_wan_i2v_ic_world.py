import argparse
import math
import os
import numpy as np
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
)
from fastvideo.utils.communications import sp_parallel_dataloader_wrapper_ic
import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    MixedPrecision
)
from torch.utils.data.distributed import DistributedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from fastvideo.dataset.latent_rl_datasets import LatentDataset, latent_image_collate_function
import torch.distributed as dist
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from fastvideo.utils.logging_ import main_print
from diffusers.video_processor import VideoProcessor
from fastvideo.utils.load import load_vae
from transformers import CLIPImageProcessor, CLIPVisionModel
from PIL import Image

from collections import deque
from diffusers.utils import export_to_video, load_image

from fastvideo.models.geometry_model import GeometryModel
from fastvideo.models.motion_model import MotionModel


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor,
    grpo: bool,
    sde_solver: bool,
):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output

    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents-pred_original_sample*(1 - sigma))/sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = ((
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2) / (2 * (std_dev_t**2))
        )
        - math.log(std_dev_t)- torch.log(torch.sqrt(2 * torch.as_tensor(math.pi))))

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean,pred_original_sample


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def encode_image(image, image_processor, image_encoder, device):
    """
    Encode image using CLIP image processor and encoder, following WAN I2V pipeline approach.
    """
    image = image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]


def run_sample_step(
    args,
    z,
    condition,
    progress_bar,
    sigma_schedule,
    transformer,
    encoder_hidden_states,
    encoder_attention_mask,
    grpo_sample,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    image_embeds=None,
    latents_mean=None,
    latents_std=None,
):
    if grpo_sample:
        all_latents = [z]
        all_log_probs = []
        for i in progress_bar:  # Add progress bar
            B = encoder_hidden_states.shape[0]
            sigma = sigma_schedule[i]
            #dsigma = sigma_schedule[i + 1] - sigma
            timestep_value = int(sigma * 1000)
            timesteps = torch.full([encoder_hidden_states.shape[0]], timestep_value, device=z.device, dtype=torch.long)
            
            with torch.no_grad():
                transformer.eval()
                
                input_latents = torch.cat([z, condition], dim=1)
                timesteps = timesteps.expand(z.shape[0])
                model_pred = transformer(
                    hidden_states=input_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timesteps,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                pred = model_pred.to(torch.float32)
            
            # Use the original z (not concatenated latents) for flux_step
            z, pred_original, log_prob = flux_step(pred, z.to(torch.float32), args.eta, sigmas=sigma_schedule, index=i, prev_sample=None, grpo=True, sde_solver=True)
            z = z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        
        # Properly denormalize latents before VAE decoding (following pipeline approach)
        latents = pred_original.to(torch.float32)
        if latents_mean is not None and latents_std is not None:
            latents = latents / latents_std.to(torch.float32) + latents_mean.to(torch.float32)
        
        all_latents = torch.stack(all_latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        all_log_probs = torch.stack(all_log_probs, dim=1)  # (batch_size, num_steps, 1)
        return z, latents, all_latents, all_log_probs


def grpo_one_step(
    args,
    latents,
    pre_latents,
    encoder_hidden_states,
    encoder_attention_mask,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    transformer,
    timesteps,
    i,
    sigma_schedule,
    condition,
    image_embeds=None,
):
    B = encoder_hidden_states.shape[0]
    with torch.autocast("cuda", torch.bfloat16):
        transformer.train()
        if args.cfg_infer > 1:
            condition_cfg = condition.repeat( (2,) + (1,)*(latents.dim()-1) )
            latent_z = latents.repeat( (2,) + (1,)*(latents.dim()-1) )
            input_latents = torch.cat([latent_z, condition_cfg], dim=1)
            
            # Prepare image embeddings for CFG
            image_embeds_cfg = None
            if image_embeds is not None:
                image_embeds_cfg = image_embeds.repeat(2, 1, 1)
            
            model_pred= transformer(
                hidden_states=input_latents,
                encoder_hidden_states=torch.cat((encoder_hidden_states,empty_cond_hidden_states),dim=0),
                timestep=timesteps.repeat( (2,) + (1,)*(timesteps.dim()-1) ),
                encoder_hidden_states_image=image_embeds_cfg,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            model_pred, uncond_pred = model_pred.chunk(2)
            pred  =  uncond_pred.to(torch.float32) + args.cfg_infer * (model_pred.to(torch.float32) - uncond_pred.to(torch.float32))
        else:
            input_latents = torch.cat([latents, condition], dim=1)
            pred= transformer(
                hidden_states=input_latents,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
    # Use the original latents (not concatenated input) for flux_step
    z, pred_original, log_prob = flux_step(pred, latents.to(torch.float32), args.eta, sigma_schedule, i, prev_sample=pre_latents.to(torch.float32), grpo=True, sde_solver=True)
    return log_prob


def sample_reference_model(
    args,
    step,
    device, 
    transformer,
    pipe_flux,
    pipe_fill,
    vae,
    encoder_hidden_states, 
    encoder_attention_mask,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    inferencer,
    caption,
    image_processor,
    image_encoder,
    image_path,
    geometry_model=None,
    motion_model=None,
):
    video_processor = VideoProcessor(vae_scale_factor=8)

    w, h, t = args.w, args.h, args.t
    sample_steps = args.sampling_steps
    sigma_schedule = torch.linspace(1, 0, args.sampling_steps + 1)
    sigma_schedule = sd3_time_shift(args.shift, sigma_schedule)

    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = encoder_hidden_states.shape[0]
    
    # Get VAE scale factors from config (like pipeline)
    vae_scale_factor_temporal = vae.config.scale_factor_temporal if hasattr(vae.config, 'scale_factor_temporal') else 4
    vae_scale_factor_spatial = vae.config.scale_factor_spatial if hasattr(vae.config, 'scale_factor_spatial') else 8
    IN_CHANNELS = vae.config.z_dim if hasattr(vae.config, 'z_dim') else 16
    
    # Validate temporal dimension compatibility (like pipeline)
    if t % vae_scale_factor_temporal != 1:
        raise ValueError(f"Video length t={t} must be compatible with vae_scale_factor_temporal={vae_scale_factor_temporal} (t % vae_scale_factor_temporal should equal 1)")
    
    latent_t = ((t - 1) // vae_scale_factor_temporal) + 1
    latent_w, latent_h = w // vae_scale_factor_spatial, h // vae_scale_factor_spatial

    batch_size = 1  
    batch_indices = torch.chunk(torch.arange(B), B // batch_size)

    vae.enable_tiling()
    all_latents = []
    all_log_probs = []
    all_rewards = [] 
    grpo_sample=True

    image = Image.open(image_path[0]).convert('RGB')
    image = image.resize((args.w, args.h))
        
    # if dist.get_rank() == 0:
    #     os.makedirs("./videos", exist_ok=True)
    # dist.barrier()

    img_save_path = os.path.join(args.output_dir, f"./videos/first_frame_rank_{dist.get_rank()}.jpg")

    # img_save_path = f"./videos/first_frame_rank_{dist.get_rank()}.jpg"
    image.save(img_save_path)

    image = load_image(img_save_path)
    image = image.resize((args.w, args.h))
    image_preprocess = video_processor.preprocess(image, height=args.h, width=args.w).to(device, dtype=torch.float32)

    # Create video condition following WAN I2V pipeline approach
    image_expanded = image_preprocess.unsqueeze(2)  # [batch_size, channels, 1, height, width]
    
    # Create video condition by padding the image to match video length
    video_condition = torch.cat([
        image_expanded, 
        image_expanded.new_zeros(image_expanded.shape[0], image_expanded.shape[1], args.t - 1, args.h, args.w)
    ], dim=2)
    
    video_condition = video_condition.to(device=device, dtype=vae.dtype)
    
    # Get VAE latents mean and std for normalization
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, dtype=torch.bfloat16)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, dtype=torch.bfloat16
    )
    
    with torch.no_grad():
        # Encode video condition and normalize
        latent_condition = vae.encode(video_condition).latent_dist.sample()
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)
        latent_condition = latent_condition.to(torch.bfloat16)
        latent_condition = (latent_condition - latents_mean) * latents_std
    
    # Get actual latent dimensions from encoded condition (may differ slightly from calculated dimensions)
    actual_latent_t, actual_latent_h, actual_latent_w = latent_condition.shape[2:]
    
    # Create mask for first frame (following pipeline approach exactly)
    mask_lat_size = torch.ones(batch_size, 1, args.t, actual_latent_h, actual_latent_w, device=device)
    mask_lat_size[:, :, list(range(1, args.t))] = 0  # Only first frame is 1, rest are 0
    
    # Process mask exactly like pipeline
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, actual_latent_h, actual_latent_w)
    mask_lat_size = mask_lat_size.transpose(1, 2)
    mask_lat_size = mask_lat_size.to(latent_condition.device)
    
    # Create final condition by concatenating mask and full latent condition (like pipeline)
    condition = torch.concat([mask_lat_size, latent_condition], dim=1)
    
    # Update latent dimensions to match actual encoded dimensions
    latent_t, latent_h, latent_w = actual_latent_t, actual_latent_h, actual_latent_w

    # Generate image embeddings if transformer supports them
    image_embeds = None
    if image_processor is not None and image_encoder is not None:
        # Check if transformer supports image embeddings
        if hasattr(transformer.config, 'image_dim') and transformer.config.image_dim is not None:
            image_embeds = encode_image(image, image_processor, image_encoder, device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(torch.bfloat16)

    if args.init_same_noise or args.use_same_noise:
        input_latents = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（1, c,t,h,w)
            device=device,
            dtype=torch.bfloat16,
        )
    for index, batch_idx in enumerate(batch_indices):        
        # Clear CUDA cache before generation to maximize available memory for FSDP operations
        torch.cuda.empty_cache()
        
        batch_encoder_hidden_states = encoder_hidden_states[batch_idx]
        batch_encoder_attention_mask = encoder_attention_mask[batch_idx]
        batch_caption = [caption[i] for i in batch_idx]
        grpo_sample = True
        progress_bar = tqdm(range(0, sample_steps), desc="Sampling Progress")
        if not (args.init_same_noise or args.use_same_noise):
            input_latents = torch.randn(
                (1, IN_CHANNELS, latent_t, latent_h, latent_w),  #（1, c,t,h,w)
                device=device,
                dtype=torch.bfloat16,
            )

        with torch.no_grad():
            z, latents, batch_latents, batch_log_probs = run_sample_step(
                args,
                input_latents.clone(),
                condition.clone(),
                progress_bar,
                sigma_schedule,
                transformer,
                batch_encoder_hidden_states,
                batch_encoder_attention_mask,
                grpo_sample,
                empty_cond_hidden_states,
                empty_cond_attention_mask,
                image_embeds,
                latents_mean,
                latents_std,
            )

        all_latents.append(batch_latents)
        all_log_probs.append(batch_log_probs)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                video = vae.decode(latents, return_dict=False)[0]
                videos = video_processor.postprocess_video(video)
        rank = int(os.environ["RANK"])

        export_to_video(videos[0], os.path.join(args.output_dir, f"videos/video_rank_{rank}_{index}.mp4"), fps=args.fps)

        # Compute rewards from enabled reward models
        with torch.no_grad():
            absolute_path = os.path.abspath(os.path.join(args.output_dir, f"videos/video_rank_{rank}_{index}.mp4"))
            reward_components = []

            # Pi3 geometry reward TODO
            if getattr(args, 'use_geometry_reward', False) and geometry_model is not None:
                geometry_reward = geometry_model.from_video_path(absolute_path)
                # Ensure reward is a CUDA tensor
                if not isinstance(geometry_reward, torch.Tensor):
                    geometry_reward = torch.tensor(geometry_reward, device=device)
                else:
                    geometry_reward = geometry_reward.to(device)
                reward_components.append(('geometry_reward', geometry_reward, getattr(args, 'geometry_reward_weight', 1.0)))

            # SpatialTracker motion consistency reward
            if getattr(args, 'use_motion_reward', False) and motion_model is not None:                
                # Compute reward
                motion_reward = motion_model.from_video_path(absolute_path)
                
                # Ensure reward is on correct device before moving models back to CPU
                if not isinstance(motion_reward, torch.Tensor):
                    motion_reward = torch.tensor(motion_reward, device=device)
                else:
                    motion_reward = motion_reward.to(device)
                reward_components.append(('motion_reward', motion_reward, getattr(args, 'motion_reward_weight', 1.0)))
                    
            # Combine rewards
            if len(reward_components) > 0:
                # Print individual reward components for debugging
                if dist.get_rank() % 8 == 0:
                    for name, val, weight in reward_components:
                        main_print(f"  {name}: {val.item():.6f} (weight: {weight:.2f})")

                # Weighted sum of rewards
                reward = sum(val * weight for name, val, weight in reward_components)
                all_rewards.append(reward.unsqueeze(0))
            else:
                # No reward models enabled, use zero reward
                all_rewards.append(torch.tensor(0.0, device=device).unsqueeze(0))

    all_latents = torch.cat(all_latents, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)
    all_rewards = torch.cat(all_rewards, dim=0)
    
    return videos, z, all_rewards, all_latents, all_log_probs, sigma_schedule, condition, image_embeds


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def train_one_step(
    args,
    device, 
    transformer,
    pipe_flux,
    pipe_fill,
    vae,
    inferencer,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    step,
    empty_cond_hidden_states,
    empty_cond_attention_mask,
    image_processor,
    image_encoder,
    geometry_model=None,
    motion_model=None,
):
    total_loss = 0.0
    grad_norm = torch.tensor(0.0, device=device)

    # Ensure model is in train mode before PPO updates
    try:
        transformer.train()
    except Exception:
        pass
    optimizer.zero_grad()
    (
        encoder_hidden_states,
        encoder_attention_mask,
        caption,
        image_path
    ) = next(loader)
    #device = latents.device
    if args.use_group:
        def repeat_tensor(tensor):
            if tensor is None:
                return None
            return torch.repeat_interleave(tensor, args.num_generations, dim=0)

        encoder_hidden_states = repeat_tensor(encoder_hidden_states)
        encoder_attention_mask = repeat_tensor(encoder_attention_mask)

        if isinstance(caption, str):
            caption = [caption] * args.num_generations
        elif isinstance(caption, list):
            caption = [item for item in caption for _ in range(args.num_generations)]
        else:
            raise ValueError(f"Unsupported caption type: {type(caption)}")

    empty_cond_hidden_states = empty_cond_hidden_states.unsqueeze(0)
    empty_cond_attention_mask = empty_cond_attention_mask.unsqueeze(0)
    videos, latents, reward, all_latents, all_log_probs, sigma_schedule, condition, image_embeds = sample_reference_model(
            args,
            step,
            device, 
            transformer,
            pipe_flux,
            pipe_fill,
            vae,
            encoder_hidden_states, 
            encoder_attention_mask, 
            empty_cond_hidden_states,
            empty_cond_attention_mask,
            inferencer,
            caption,
            image_processor,
            image_encoder,
            image_path,
            geometry_model=geometry_model,
            motion_model=motion_model,
        )
    batch_size = all_latents.shape[0]
    timestep_value = [int(sigma * 1000) for sigma in sigma_schedule][:args.sampling_steps]
    timestep_values = [timestep_value[:] for _ in range(batch_size)]
    device = all_latents.device
    timesteps =  torch.tensor(timestep_values, device=all_latents.device, dtype=torch.long)
    samples = {
        "timesteps": timesteps.detach().clone()[:, :-1],
        "latents": all_latents[
            :, :-1
        ][:, :-1],  # each entry is the latent before timestep t
        "next_latents": all_latents[
            :, 1:
        ][:, :-1],  # each entry is the latent after timestep t
        "log_probs": all_log_probs[:, :-1],
        "rewards": reward.to(torch.float32),
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "empty_cond_hidden_states": empty_cond_hidden_states.repeat(batch_size, 1, 1),
        "empty_cond_attention_mask": empty_cond_attention_mask.repeat(batch_size, 1),
        "image_embeds": image_embeds.repeat(batch_size, 1, 1) if image_embeds is not None else None,
        "condition": condition.repeat(batch_size, 1, 1, 1, 1) if condition is not None else None,
    }
    gathered_reward = gather_tensor(samples["rewards"])
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open(os.path.join(args.output_dir, 'reward.txt'), 'a') as f: 
            f.write(f"{gathered_reward.mean().item()}\n")

    #calculate advantage
    if args.use_group:
        n = len(samples["rewards"]) // (args.num_generations)
        advantages = torch.zeros_like(samples["rewards"])
        
        for i in range(n):
            start_idx = i * args.num_generations
            end_idx = (i + 1) * args.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        
        samples["advantages"] = advantages
    else:
        advantages = (samples["rewards"] - gathered_reward.mean())/(gathered_reward.std()+1e-8)
        samples["advantages"] = advantages
    
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device) 
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    # Create a list of samples, one dict per sample in the batch
    # This allows us to iterate over each sample separately for gradient accumulation
    samples_batched_list = [
        {k: v[i:i+1] if v is not None else None for k, v in samples.items()}
        for i in range(batch_size)
    ]
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)

    # Select a representative parameter to track updates
    # Prefer LoRA parameters if using LoRA, otherwise pick a larger parameter
    sample_param = None
    sample_param_name = None
    for name, p in transformer.named_parameters():
        if p.requires_grad and p.data is not None and p.numel() > 0:
            # Prefer LoRA parameters or larger parameters
            if sample_param is None or 'lora_' in name or p.numel() > sample_param.numel():
                sample_param = p
                sample_param_name = name
                if 'lora_' in name:  # Stop at first LoRA param if found
                    break

    for i,sample in list(enumerate(samples_batched_list)):
        for _ in range(train_timesteps):
            clip_range = 0.1  # Increased from 1e-4 to allow more meaningful updates
            adv_clip_max = 5.0
            new_log_probs = grpo_one_step(
                args,
                sample["latents"][:,_],
                sample["next_latents"][:,_],
                sample["encoder_hidden_states"],
                sample["encoder_attention_mask"],
                sample["empty_cond_hidden_states"],
                sample["empty_cond_attention_mask"],
                transformer,
                sample["timesteps"][:,_],
                perms[i][_],
                sigma_schedule,
                sample["condition"],
                sample["image_embeds"],
            )

            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )

            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])

            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            # Match train_grpo_skyreels_i2v.py: divide by both gradient_accumulation_steps and train_timesteps
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)

            # Need retain_graph=True for all but the last backward pass
            # because samples share tensors (condition, image_embeds) in the computation graph
            is_last_sample = (i == len(samples_batched_list) - 1)
            is_last_timestep = (_ == train_timesteps - 1)
            should_retain = not (is_last_sample and is_last_timestep)
            loss.backward(retain_graph=should_retain)
            
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            total_loss += avg_loss.item()
        
        # Optimizer step after accumulating gradients
        if (i+1) % args.gradient_accumulation_steps == 0:            
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        if dist.get_rank() % 8 == 0:
            print("reward", sample["rewards"].item())
            print("advantage", sample["advantages"].item())
            print("final loss", loss.item())
        dist.barrier()
    
    return total_loss, grad_norm.item()


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    initialize_sequence_parallel_state(args.sp_size)

    # We use different seeds for the noise generation in each process to ensure that the noise is different in a batch.
    noise_random_generator = None

    # Handle the repository creation
    if rank <= 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "videos"), exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inferencer = None
    geometry_model = None
    motion_model = None

    # Initialize Pi3 if needed
    if args.use_geometry_reward:
        geometry_model = GeometryModel(
            device=device, 
            interval=args.interval, 
            pi3_model_path=args.pi3_model_path,
            lepard_config_path=args.lepard_config_path,
            lepard_model_path=args.lepard_model_path
        )
    
    # Initialize SpatialTracker if needed
    if args.use_motion_reward:
        motion_model = MotionModel(
            device=device,
            interval=args.interval,
            grid_size=args.motion_grid_size,
            vo_points=args.motion_vo_points,
            tracker_model_path=args.tracker_model_path,
            vggt_model_path=args.vggt_model_path
        )

    main_print(f"--> loading model from {args.pretrained_model_name_or_path}")
    main_print(f"--> loading model from {args.model_type}")
    
    transformer = load_transformer(
        args.model_type,
        args.dit_model_name_or_path,
        args.pretrained_model_name_or_path,
        torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16,
    )

    # Set up LoRA if enabled
    if args.use_lora:
        main_print("--> Setting up LoRA training")
        
        # Define target modules for LoRA based on model type
        if args.model_type == "hunyuan_hf":
            target_modules = ["to_k", "to_q", "to_v", "to_out.0", "mlp.fc1", "mlp.fc2"]
        else:
            # Default target modules for other model types
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        
        transformer.requires_grad_(False)
        transformer_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)
        
        # Store LoRA config in transformer for checkpoint saving
        transformer.config.lora_rank = args.lora_rank
        transformer.config.lora_alpha = args.lora_alpha
        transformer.config.lora_target_modules = target_modules
        
        # Load pre-trained LoRA weights if specified (before FSDP wrapping)
        if args.load_lora_weights:
            main_print(f"--> Loading LoRA weights from: {args.load_lora_weights}")
            lora_state_dict = None
            
            # Try loading safetensors first
            safetensors_path = os.path.join(args.load_lora_weights, "pytorch_lora_weights.safetensors")
            if os.path.exists(safetensors_path):
                try:
                    from safetensors.torch import load_file
                    lora_state_dict = load_file(safetensors_path)
                    main_print("--> LoRA weights loaded successfully from safetensors")
                except Exception as e:
                    main_print(f"--> Warning: Could not load safetensors file: {e}")
            
            # Load the state dict if we successfully loaded it
            if lora_state_dict is not None:
                try:
                    # Get current LoRA parameter statistics before loading
                    lora_params_before = {}
                    for name, param in transformer.named_parameters():
                        if 'lora_' in name:
                            lora_params_before[name] = {
                                'mean': param.data.mean().item(),
                                'std': param.data.std().item(),
                                'norm': param.data.norm().item(),
                            }
                    
                    # Apply the loaded state dict
                    set_peft_model_state_dict(transformer, lora_state_dict)
                    main_print("--> LoRA state dict applied to model successfully")
                    
                    # Validate that parameters actually changed
                    main_print("--> Validating LoRA weights were loaded correctly:")
                    num_changed = 0
                    num_total = 0
                    max_samples = 3  # Show stats for first 3 LoRA parameters
                    
                    for name, param in transformer.named_parameters():
                        if 'lora_' in name:
                            num_total += 1
                            current_norm = param.data.norm().item()
                            
                            # Check if parameter changed from initialization
                            if name in lora_params_before:
                                prev_norm = lora_params_before[name]['norm']
                                if abs(current_norm - prev_norm) > 1e-6:
                                    num_changed += 1
                            
                            # Print detailed stats for first few parameters
                            if num_total <= max_samples:
                                main_print(f"    {name}:")
                                main_print(f"      - Shape: {tuple(param.shape)}")
                                main_print(f"      - Mean: {param.data.mean().item():.6e}")
                                main_print(f"      - Std: {param.data.std().item():.6e}")
                                main_print(f"      - Norm: {current_norm:.6e}")
                                if name in lora_params_before:
                                    main_print(f"      - Changed: {'Yes' if abs(current_norm - lora_params_before[name]['norm']) > 1e-6 else 'No'}")
                    
                    main_print(f"--> Validation summary: {num_changed}/{num_total} LoRA parameters changed")
                    
                    if num_changed == 0:
                        main_print("--> WARNING: No LoRA parameters changed! Checkpoint may not have been loaded correctly.")
                    elif num_changed < num_total:
                        main_print(f"--> WARNING: Only {num_changed}/{num_total} parameters changed. Some weights may not have loaded.")
                    else:
                        main_print("--> ✓ All LoRA parameters successfully loaded and validated")
                    
                except Exception as e:
                    main_print(f"--> Warning: Could not apply LoRA state dict: {e}")
                    main_print("--> Continuing with fresh LoRA initialization")
            else:
                main_print(f"--> Warning: No valid LoRA checkpoint found at {args.load_lora_weights}")
                main_print("--> Continuing with fresh LoRA initialization")

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )

    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
    )

    # Prevent FSDP from wrapping condition_embedder to keep parameters accessible
    # This fixes the StopIteration error when accessing time_embedder parameters
    # We'll manually move it to device after FSDP wrapping
    fsdp_kwargs["ignored_modules"] = [transformer.condition_embedder]

    if args.use_lora:
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        )

        if args.master_weight_type.lower() == "fp32":
            master_dtype = torch.float32
        else:
            master_dtype = torch.bfloat16

        # Make ALL params & buffers uniform, including LoRA
        transformer.to(master_dtype)
        for b in transformer.buffers():
            if b.dtype in (torch.float32, torch.bfloat16) and b.dtype != master_dtype:
                b.data = b.data.to(master_dtype)

        # PEFT LoRA sometimes leaves A/B in fp32 — hard cast them too
        for n, p in transformer.named_parameters():
            if "lora_" in n and p.dtype != master_dtype:
                p.data = p.data.to(master_dtype)

        transformer._no_split_modules = [
            no_split_module.__name__ for no_split_module in no_split_modules
        ]
        fsdp_kwargs["auto_wrap_policy"] = fsdp_kwargs["auto_wrap_policy"](transformer)
        
        
    transformer = FSDP(transformer, **fsdp_kwargs,)

    # Move ignored modules to device manually since FSDP won't do it
    # This is necessary because condition_embedder is in ignored_modules
    target_dtype = torch.float32 if args.master_weight_type == "fp32" else torch.bfloat16
    transformer._fsdp_wrapped_module.condition_embedder.to(device=device, dtype=target_dtype)

    # Load LoRA checkpoint if resuming (will be called after optimizer is created)
    init_steps = 0
    if args.resume_from_lora_checkpoint and args.use_lora:
        main_print(f"--> Will resume from LoRA checkpoint: {args.resume_from_lora_checkpoint}")
        init_steps = int(args.resume_from_lora_checkpoint.split("-")[-2]) if "-" in args.resume_from_lora_checkpoint else 0

    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(
            transformer, no_split_modules, args.selective_checkpointing
        )
    
    # Set model as trainable.
    transformer.train()


    # Initialize image encoder and processor for WAN I2V
    image_processor = None
    image_encoder = None
    
    # Check if transformer supports image embeddings
    if hasattr(transformer.config, 'image_dim') and transformer.config.image_dim is not None:
        main_print("--> Loading image encoder and processor for WAN I2V")
        # Load image processor and encoder from the model path
        image_processor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="image_processor"
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="image_encoder",
            torch_dtype=torch.float32
        ).to(device)
        main_print("--> Image encoder and processor loaded successfully")
    else:
        main_print("--> Transformer does not support image embeddings, skipping image encoder/processor")


    '''
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        pipe_flux.transformer,
        args.fsdp_sharding_startegy,
        False,
        args.use_cpu_offload,
        args.master_weight_type,
    )
    
    pipe_flux.transformer = FSDP(pipe_flux.transformer, **fsdp_kwargs,).eval()
    pipe_flux.vae.to(device)
    pipe_flux.text_encoder.to(device)
    '''
    # Diagnostics: total vs trainable params
    total_params_count = sum(p.numel() for p in transformer.parameters())
    trainable_params_count = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    main_print(f"Total params: {total_params_count:,}; trainable: {trainable_params_count:,}")

    if args.use_lora:
        # Only optimize LoRA parameters
        params_to_optimize = []
        for name, param in transformer.named_parameters():
            if 'lora_' in name and param.requires_grad:
                params_to_optimize.append(param)
        
        # Double-check that we only have LoRA parameters
        non_lora_params = [p for p in params_to_optimize if 'lora_' not in next((n for n, param in transformer.named_parameters() if param is p), '')]
        if non_lora_params:
            main_print(f"--> Warning: Found {len(non_lora_params)} non-LoRA parameters in optimizer!")
        
        main_print(f"LoRA params passed to optimizer: {sum(p.numel() for p in params_to_optimize):,} across {len(params_to_optimize)} tensors")
    else:
        # Optimize all trainable parameters
        params_to_optimize = transformer.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
        main_print(f"Params passed to optimizer: {sum(p.numel() for p in params_to_optimize):,} across {len(params_to_optimize)} tensors")

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    try:
        main_print(f"Optimizer groups: {len(optimizer.param_groups)}, lr={optimizer.param_groups[0].get('lr', None)}")
    except Exception:
        pass

    main_print(f"optimizer: {optimizer}")

    # Load LoRA checkpoint if resuming
    if args.resume_from_lora_checkpoint and args.use_lora:
        main_print(f"--> Loading LoRA checkpoint: {args.resume_from_lora_checkpoint}")
        try:
            resume_lora_optimizer(transformer, args.resume_from_lora_checkpoint, optimizer)
            main_print("--> LoRA checkpoint loaded successfully")
        except Exception as e:
            main_print(f"--> Warning: Could not load LoRA checkpoint: {e}")
            main_print("--> Continuing with fresh LoRA initialization")

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    train_dataset = LatentDataset(args.data_json_path, args.num_latent_t, args.cfg)
    sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed
        )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=latent_image_collate_function,
        pin_memory=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader)
        / args.gradient_accumulation_steps
        * args.sp_size
        / args.train_sp_batch_size
    )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    vae, autocast_type, fps = load_vae(args.model_type, args.vae_model_path)
    #vae.enable_tiling()

    if rank <= 0:
        project = args.tracker_project_name or "fastvideo"
        wandb.init(project=project, config=args)

    # Train!
    total_batch_size = (
        world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(
        f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}"
    )
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B"
    )
    # print dtype
    main_print(f"  Master weight dtype: {transformer.parameters().__next__().dtype}")

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError("resume_from_checkpoint is not supported now.")
        # TODO

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    loader = sp_parallel_dataloader_wrapper_ic(
        train_dataloader,
        device,
        args.train_batch_size,
        args.sp_size,
        args.train_sp_batch_size,
    )

    step_times = deque(maxlen=100)
    empty_cond_hidden_states = torch.load(
        "./data/empty/prompt_embed/0.pt", map_location=torch.device(f'cuda:{device}'),weights_only=True
    )
    empty_cond_attention_mask = torch.load(
        "./data/empty/prompt_attention_mask/0.pt", map_location=torch.device(f'cuda:{device}'),weights_only=True
    )
        

    for epoch in range(1):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch
        for step in range(init_steps+1, args.max_train_steps+1):
            start_time = time.time()
            if step % args.checkpointing_steps == 0:
                if args.use_lora:
                    # For LoRA, use a simpler checkpoint saving approach without pipeline
                    save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, step, None, epoch)
                else:
                    save_checkpoint(transformer, rank, args.output_dir, step, epoch)

                dist.barrier()
            loss, grad_norm = train_one_step(
                args,
                device, 
                transformer,
                None,
                None,
                vae,
                inferencer,
                optimizer,
                lr_scheduler,
                loader,
                args.max_grad_norm,
                step,
                empty_cond_hidden_states,
                empty_cond_attention_mask,
                image_processor,
                image_encoder,
                geometry_model=geometry_model,
                motion_model=motion_model,
            )
    
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            if rank <= 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step_time": step_time,
                        "avg_step_time": avg_step_time,
                        "grad_norm": grad_norm,
                    },
                    step=step,
                )
    
    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    from lightning.pytorch import seed_everything
    seed_everything(42)


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="hunyuan_hf", help="The type of model to train."
    )
    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_latent_t", type=int, default=28, help="Number of latent timesteps."
    )
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO

    # text encoder & vae & diffusion model
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )

    # logs
    parser.add_argument("--tracker_project_name", type=str, default=None)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_lora_weights",
        type=str,
        default=None,
        help=(
            "Path to a LoRA checkpoint to load weights from (without optimizer state). "
            "Use this to start training from pre-trained LoRA weights."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=1, help="Alpha parameter for LoRA."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=64, help="LoRA rank parameter. "
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,   
        help="Reward model path",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=None,   
        help="video height",
    )
    parser.add_argument(
        "--w",
        type=int,
        default=None,   
        help="video width",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=None,   
        help="video length",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,   
        help="fps of stored video",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_geometry_reward",
        action="store_true",
        default=False,
        help="Enable Pi3-based geometry consistency reward on 1x2 grid videos",
    )
    parser.add_argument(
        "--geometry_reward_weight",
        type=float,
        default=1.0,
        help="Scale factor applied to negative MSE distance for reward",
    )
    parser.add_argument(
        "--pi3_model_path",
        type=str,
        default=None,
        help="Path to Pi3 model for geometry reward",
    )
    parser.add_argument(
        "--lepard_config_path",
        type=str,
        default=None,
        help="Path to LePARD config for geometry reward",
    )
    parser.add_argument(
        "--lepard_model_path",
        type=str,
        default=None,
        help="Path to LePARD model for geometry reward",
    )
    parser.add_argument(
        "--use_motion_reward",
        action="store_true",
        default=False,
        help="Enable SpatialTracker-based motion consistency reward on 1x2 grid videos",
    )
    parser.add_argument(
        "--motion_reward_weight",
        type=float,
        default=1.0,
        help="Weight factor for SpatialTracker reward",
    )
    parser.add_argument(
        "--motion_grid_size",
        type=int,
        default=10,
        help="Grid size for SpatialTracker query points",
    )
    parser.add_argument(
        "--motion_vo_points",
        type=int,
        default=756,
        help="Number of tracking points in SpatialTracker",
    )
    parser.add_argument(
        "--tracker_model_path",
        type=str,
        default=None,
        help="Path to tracker model for motion reward",
    )
    parser.add_argument(
        "--vggt_model_path",
        type=str,
        default=None,
        help="Path to VGGT model for motion reward",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Frame sampling interval for input sequences",
    )
    parser.add_argument(
        "--init_same_noise",
        action="store_true",
        default=False,
        help="whether to use the same noise",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep_fraction",
    )
    parser.add_argument(
        "--cfg_infer",
        type = float,
        default=1.0,
        help="cfg",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="sampling shift",
    )
    parser.add_argument(
        "--use_same_noise",
        action="store_true",
        default=False,
        help="whether to use the same noise for all samples",
    )

    args = parser.parse_args()
    main(args)
