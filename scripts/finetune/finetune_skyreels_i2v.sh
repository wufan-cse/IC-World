GPU_NUM=8 # 2,4,8
MODEL_PATH="/dockerdata/wufan/SkyReels-V1-I2V"
OUTPUT_DIR="data/rl_embeddings"

cp -rf /dockerdata/wufan/SkyReels-V1-I2V/tokenizer/* /dockerdata/wufan/SkyReels-V1-I2V/text_encoder
cp -rf /dockerdata/wufan/SkyReels-V1-I2V/tokenizer_2/* /dockerdata/wufan/SkyReels-V1-I2V/text_encoder_2

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/grid-ic_1000.txt" \
    --model_type hunyuan_hf


GPU_NUM=8 # 2,4,8
MODEL_PATH="/dockerdata/wufan/SkyReels-V1-I2V"
OUTPUT_DIR="data/empty"

torchrun --nproc_per_node=$GPU_NUM --master_port 19003 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/empty.txt" \
    --model_type hunyuan_hf



export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export HF_HOME="/dockerdata/wufan/DanceGRPO/ckpts"
# export TRANSFORMERS_CACHE="/mnt/kaiwu-group-y1-sh-hdd/wufan/codes/DanceGRPO/ckpts"
# export TORCH_HOME="/mnt/kaiwu-group-y1-sh-hdd/wufan/codes/DanceGRPO/ckpts"

# pip3 install moviepy
# mkdir videos
# pip3 install huggingface_hub==0.24.0 
# pip3 install huggingface-hub==0.34.4                   
# pip3 install tf-keras==2.19.0
# pip3 install trl==0.16.0
# pip3 install transformers==4.46.1
# pip3 install protobuf==5.29.5

###Actually, we don't use the original pytorch torchrun in our internal environment, 
###so I just follow the official example of pytorch.
###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
    fastvideo/train_grpo_skyreels_i2v.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path /dockerdata/wufan/SkyReels-V1-I2V \
    --reference_model_path /dockerdata/wufan/FLUX.1-dev \
    --reference_fill_path /dockerdata/wufan/Qwen-Image-Edit \
    --vae_model_path /dockerdata/wufan/SkyReels-V1-I2V \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 121 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 5 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 1 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo \
    --tracker_project_name skyreels_i2v \
    --h 400 \
    --w 640 \
    --t 53 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --cfg_infer 5.0 \
    --shift 7 \
    --use_group \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --use_pi3_geometry_reward --pi3_mode scene --pi3_interval 10 --pi3_reward_weight 1.0 \
    # --max_train_steps 121 \
    # --checkpointing_steps 40 \
    # --gradient_accumulation_steps 8 \
    # --num_generations 8 \
    # --use_videoalign \


python /mnt/kaiwu-group-y1-sh-hdd/wufan/codes/warm_gpu.py --size 60000 --gpus 8 --interval 0.01
