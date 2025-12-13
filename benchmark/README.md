# Evaluation

We provide the evaluation code of our proposed geometry consistency and motion consistency metrics for evaluating shared world videos.
For a detailed explanation of these metrics, please refer to our [paper](https://arxiv.org/abs/2512.02793).


1. **Weights Preparation:**

    LEPARD model can be downloaded [here](https://drive.google.com/file/d/17QGX_wwtDPXN1GSKJHY-6RTIRPz90RLn/view?usp=sharing).

    Pi3 model can be downloaded using huggingface, please refer [here](https://huggingface.co/yyfz233/Pi3).

    SpatialTrackerV2 model contains two parts, [Front model](https://huggingface.co/Yuxihenry/SpatialTrackerV2_Front) and [Offline model](https://huggingface.co/Yuxihenry/SpatialTrackerV2-Offline).

    Downloaded files can be structured like this:

    ```bash
    IC-World/weights
        |-- lepard/pretrained/3dmatch/model_best_loss.pth
        |-- Pi3
        |   |-- config.json
        |   |-- model.safetensors
        |-- SpatialTrackerV2_Front
        |   |-- config.json
        |   |-- model.safetensors
         -- SpatialTrackerV2-Offline    
            |-- config.json
            |-- model.safetensors
    ```

2. **Environment Preparation:**


    ```bash
    # clone the code
    git clone https://github.com/wufan-cse/IC-World.git
    cd IC-World
    git submodule update --init --recursive

    # create environment
    conda create -n icworld python=3.10
    conda activate icworld

    pip install -r requirements.txt
    ```

3. **Run Evaluation from Command Line:**

    ```bash
    # cd to root directory
    cd IC-World

    # calculate the geometry consistency score
    python benchmark/geometry_evaluator.py \
        --video_dir <your_own_directory> \
        --confidence_threshold 0.1 \
        --interval 10

    # calculate the motion consistency score
    python benchmark/motion_evaluator.py \
        --video_dir <your_own_directory> \
        --grid_size 10 \
        --interval 10
    ```

    **Arguments:**

    * `--video_dir`: Path to the input video directory. **Note that each video is a horizontal combination of two sub-video**. (Default: `assets`)
    * `--confidence_threshold`: Confidence threshold for point filtering (choose from: `0.1`, `0.5`, `0.7`). (Default: `0.1`)
    * `--grid_size`: Grid size of query points(choose from: `10`, `20`, `30`). (Default: `10`)
    * `--interval`: Frame sampling interval. (Default: `10`)
    <!-- * `--device`: Device to run inference on. (Default: `cuda`) -->
