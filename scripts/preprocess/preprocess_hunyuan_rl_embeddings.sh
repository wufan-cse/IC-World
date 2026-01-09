GPU_NUM=8 # 2,4,8
MODEL_PATH="/dockerdata/wufan/HunyuanVideo"
OUTPUT_DIR="data/rl_embeddings"

cp -rf /dockerdata/wufan/HunyuanVideo/tokenizer/* /dockerdata/wufan/HunyuanVideo/text_encoder
cp -rf /dockerdata/wufan/HunyuanVideo/tokenizer_2/* /dockerdata/wufan/HunyuanVideo/text_encoder_2

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_hunyuan_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/video_prompts_debug.txt" \
    --model_type hunyuan_hf
