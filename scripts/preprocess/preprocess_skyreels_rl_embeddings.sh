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
