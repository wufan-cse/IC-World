VIDEO_DIR="./assets/benchmark"

python fastvideo/models/geometry_model.py \
    --video_dir $VIDEO_DIR \
    --confidence_threshold 0.1 \
    --interval 5