export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
# export WANDB_API_KEY="4d9a0949fe32ea546a6ce7609b8bcede2beca295"
export MODEL_PATH="/dockerdata/wufan/HunyuanVideo"
export HF_HOME="/dockerdata/wufan/DanceGRPO/ckpts"

# pip3 install moviepy
# mkdir videos
# pip3 install huggingface_hub==0.24.0 
# pip3 install tf-keras==2.19.0
# pip3 install trl==0.16.0
# pip3 install transformers==4.46.1
# pip3 install protobuf==5.29.5

###Actually, we don't use the original pytorch torchrun in our internal environment, 
###so I just follow the official example of pytorch.
###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
    fastvideo/train_grpo_hunyuan.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path $MODEL_PATH \
    --vae_model_path $MODEL_PATH \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --validation_prompt_dir data/Mochi-Black-Myth/validation \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 5 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --validation_steps 100000000 \
    --validation_sampling_steps 8 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir data/outputs/grpo \
    --tracker_project_name grpo \
    --h 480 \
    --w 480 \
    --t 53 \
    --sampling_steps 16 \
    --eta 0.25 \
    --lr_warmup_steps 0 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 2 \
    --shift 5 \
    --use_group \
    --timestep_fraction 0.6 \
    --use_same_noise \
    --bestofn 8 \
    --vq_coef 0.0 \
    --mq_coef 1.0 \
    --use_pi3_geometry_reward --pi3_mode scene --pi3_interval 10 --pi3_reward_weight 1.0 \
    # Origin settings
    # --checkpointing_steps 50 \
    # --gradient_accumulation_steps 4 \
    # --sampling_steps 16 \
    # --learning_rate 1e-5 \
    # --max_train_steps 202 \
    # --vq_coef 1.0 \
    # --mq_coef 0.0 \
    # --num_generations 24 \
    # --use_videoalign \
