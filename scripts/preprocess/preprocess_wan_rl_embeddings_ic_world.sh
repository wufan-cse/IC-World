GPU_NUM=2 # 1,2,4,8
MODEL_PATH="./weights/IC-World-I2V-14B"
OUTPUT_DIR="./data/preprocess/static_scene_dynamic_camera_train"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_wan_i2v_rl_embeddings_ic_world.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --guidance_scale 1.0 \
    --prompt_file "./data/IC-World-dataset/static_scene_dynamic_camera_train.txt" \
    --model_type wan \
    --is_static_scene

OUTPUT_DIR="./data/empty"

torchrun --nproc_per_node=$GPU_NUM --master_port 19003 \
    fastvideo/data_preprocess/preprocess_wan_i2v_rl_embeddings.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "./assets/empty.txt" \
    --model_type wan