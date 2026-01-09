VIDEO_DIR="./assets/benchmark"

python fastvideo/models/motion_model.py \
    --video_dir $VIDEO_DIR \
    --grid_size 10 \
    --interval 5