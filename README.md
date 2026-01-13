<p align="center"><img src="assets/logo.jpg" width="25%"></p>
<div align="center">
  <a href='https://arxiv.org/abs/2512.02793'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://github.com/wufan-cse/IC-World'><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp;
  </a>
</div>

This is the official implementation for [paper](https://arxiv.org/abs/2512.02793), IC-World: In-Context Generation for Shared World Modeling. This implementation is based on FastVideo & DanceGRPO, supporting advanced Wan2.1-I2V-14B with efficient multi-nodes training and 4-steps inference.

<!-- ## Key Features

IC-World has the following features:
- Support GRPO finetuning + Wan-2.1-I2V-14B -->

## Updates

- __[2026.01.10]__: 🔥 We released training & inference codes.
- __[2025.12.28]__: We released [dataset](https://huggingface.co/datasets/fffan/IC-World-dataset) used in IC-World.
- __[2025.12.13]__: We released evaluation codes.
- __[2025.12.03]__: 🔥 We released the paper in arXiv!

If you have any research or engineering inquiries, feel free to open issues or email us directly at fan011@e.ntu.edu.sg.

## Getting Started

### Weights Preparation

Our trained model can be download in [fffan/IC-World-I2V-14B](https://huggingface.co/fffan/IC-World-I2V-14B).

LEPARD model can be downloaded [here](https://drive.google.com/file/d/17QGX_wwtDPXN1GSKJHY-6RTIRPz90RLn/view?usp=sharing).

Pi3 model can be downloaded using huggingface, please refer [here](https://huggingface.co/yyfz233/Pi3).

SpatialTrackerV2 model contains two parts, [Front model](https://huggingface.co/Yuxihenry/SpatialTrackerV2_Front) and [Offline model](https://huggingface.co/Yuxihenry/SpatialTrackerV2-Offline).

```bash
IC-World/weights
    ├── IC-World-I2V-14B
    ├── lepard/pretrained/3dmatch/model_best_loss.pth
    ├── Pi3
    │   ├── config.json
    │   ├── model.safetensors
    ├── SpatialTrackerV2_Front
    │   ├── config.json
    │   ├── model.safetensors
    └── SpatialTrackerV2-Offline    
        ├── config.json
        └── model.safetensors
```

### Installation

```bash
# clone the code
git clone https://github.com/wufan-cse/IC-World.git
cd IC-World
git submodule update --init --recursive

# create environment
conda create -n icworld python=3.10
conda activate icworld

pip install -e .
```

### Inference

```bash
python inference.py \
    --pretrained_model_name_or_path ./weights/IC-World-I2V-14B \
    --lora_weights_path ./weights/IC-World-I2V-14B \
    --lora_weight_name pytorch_lora_weights.safetensors \
    --input_image1 ./assets/img.png \
    --input_image2 ./assets/img1.png \
    --prompt "" \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --fps 16 \
    --guidance_scale 1.0 \
    --num_inference_steps 4 \
    --seed 42 \
    --output output.mp4
```

### Training

Two settings, adapt the enviroment variables: 
1. static_scene_dynamic_camera_train
2. dynamic_scene_static_camera_train

```bash
# preprocessing with 8 H20 GPUs
# setup PROMPT_FILE & OUTPUT_DIR for the two settings
export PROMPT_FILE="./data/IC-World-dataset/static_scene_dynamic_camera_train.txt"
export OUTPUT_DIR="./data/preprocess/static_scene_dynamic_camera_train"

bash scripts/preprocess/preprocess_wan_rl_embeddings_ic_world.sh

# using the following script for training with 8 H20 GPUs or other GPUs with more than 80GB, such as H200
# setup DATA_JSON_PATH
export DATA_JSON_PATH="./data/preprocess/static_scene_dynamic_camera_train/videos2caption.json"
bash scripts/finetune/finetune_wan_i2v_grpo_ic_world.sh 
```


### Evaluation

More details can be found in [benchmark](https://github.com/wufan-cse/IC-World/tree/main/benchmark).

```bash
# calculate the geometry consistency score
python fastvideo/models/geometry_model.py \
    --video_dir <your_own_directory> \
    --confidence_threshold 0.1 \
    --interval 5

# calculate the motion consistency score
python fastvideo/models/motion_model.py \
    --video_dir <your_own_directory> \
    --grid_size 10 \
    --interval 5
```

**Arguments:**

* `--video_dir`: Path to the input video directory. **Note that each video is a horizontal combination of two sub-video**. (Default: `assets`)
* `--confidence_threshold`: Confidence threshold for point filtering (choose from: `0.1`, `0.5`, `0.7`). (Default: `0.1`)
* `--grid_size`: Grid size of query points(choose from: `10`, `20`, `30`). (Default: `10`)
* `--interval`: Frame sampling interval. (Default: `5`)
<!-- * `--device`: Device to run inference on. (Default: `cuda`) -->



<!-- ## Evaluation
We provide a simple inference code using Wan2.2-14B-I2V and examples that allow readers to quickly experience the core idea of this paper.

```bash
# Create and activate a new conda environment
conda create -n icworld python=3.10
conda activate icworld

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install diffusers

# Inference
python inference.py -i examples/first_frames/0.jpg -p examples/prompts/0.txt -o examples/results/0.mp4
``` -->


## Inference Demos of IC-World

<!-- Here we present some rough demos for quick look. -->

### Static scene + Dynamic camera

- In the left demo, the letter above the door first appears in the left view and later reappears in the right view. 
- In the right demo, the advertising tag on the lower-right table first appears in the right view and subsequently reappears in the left view.

<!-- <video src="https://github.com/user-attachments/assets/5a97daaf-95ce-4cbb-a332-5ba4655e7914" width="48%" controls loop></video>
<video src="https://github.com/user-attachments/assets/6a92d736-03c3-4728-b9c3-2b06a58aa09e" width="48%" controls loop></video> -->

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/5a97daaf-95ce-4cbb-a332-5ba4655e7914" width="100%" controls loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/6a92d736-03c3-4728-b9c3-2b06a58aa09e" width="100%" controls loop></video>
      </td>
  </tr>
</table>

### Dynamic scene + Static camera

<video src="https://github.com/user-attachments/assets/4dc66c86-59f8-47d6-af41-00f0700eff25" width="0.48" controls loop></video>


## TODOs

- [ ] Support more video foundation models.
- [ ] Release checkpoints (before 2026.01.31).
- [x] Release training & inference codes.
- [x] Release dataset.
- [x] Release inference codes.
- [x] Release evaluation metrics codes.
- [x] Release paper.


## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) & [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [Diffusers](https://github.com/huggingface/diffusers)

We thank the authors for their contributions to the community!

## Citation
If you find IC-World useful and insightful for your research, please consider giving a star ⭐ and citation.

```bibtex
@article{wu2025icworld,
  title={IC-World: In-Context Generation for Shared World Modeling},
  author={Wu, Fan and Wei, Jiacheng and Li, Ruibo and Xu, Yi and Li, Junyou and Ye, Deheng and Lin, Guosheng},
  journal={arXiv preprint arXiv:2512.02793},
  year={2025}
}
```
