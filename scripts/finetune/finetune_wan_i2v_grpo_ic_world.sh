export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline

RANK=0

MODEL_PATH="./weights/IC-World-I2V-14B"
# LORA_WEIGHT_PATH=None

DATA_JSON_PATH="./data/preprocess/static_scene_dynamic_camera_train/videos2caption.json"
OUTPUT_DIR="./outputs/grpo_geo_bs1_lr1e-5_gas8_ng32_lora_rank64_max101"

###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=$RANK --master_addr=127.0.0.1 --master_port=29501 \
    fastvideo/train_grpo_wan_i2v_ic_world.py \
    --model_type "wan" \
    --pretrained_model_name_or_path $MODEL_PATH \
    --vae_model_path $MODEL_PATH \
    --data_json_path $DATA_JSON_PATH \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 101 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --cfg_infer 1.0 \
    --ema_decay 0.999 \
    --output_dir $OUTPUT_DIR \
    --tracker_project_name grpo \
    --h 480 \
    --w 832 \
    --t 49 \
    --sampling_steps 4 \
    --eta 0.25 \
    --lr_warmup_steps 0 \
    --fps 16 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 32 \
    --shift 3.0 \
    --use_group \
    --timestep_fraction 0.6 \
    --use_lora \
    --lora_rank 64 \
    --master_weight_type bf16 \
    --interval 10 \
    --pi3_model_path "./weights/Pi3" \
    --lepard_config_path "./thirdparty/lepard/configs/test/3dmatch.yaml" \
    --lepard_model_path "./weights/lepard/pretrained/3dmatch/model_best_loss.pth" \
    --tracker_model_path "./weights/SpatialTrackerV2-Offline" \
    --vggt_model_path "./weights/SpatialTrackerV2_Front" \
    --motion_grid_size 10 \
    --motion_vo_points 756 \
    --use_geometry_reward \
    --geometry_reward_weight 1.0 \
    # --use_motion_reward \
    # --motion_reward_weight 1.0
    #--load_lora_weights $LORA_WEIGHT_PATH \

