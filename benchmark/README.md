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


4. **Example Results:**

    Example videos used for evaluation can be found in [assets/benchmark](https://github.com/wufan-cse/IC-World/tree/main/assets/benchmark).

    Geometry consistency score:
    - Bad example (Left): Geometry_{average} = 0.8833
    - Good example (Right): Geometry_{average} = 0.8950

    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
    <tr>
        <td>
            <video src="https://github.com/user-attachments/assets/f24247ee-217c-4927-9047-b0ac74d0532e" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/c88104ad-fbee-4da6-84ff-c090a3f1fd77" width="100%" controls loop></video>
        </td>
    </tr>
    </table>

    Motion consistency score:
    - Bad example (Left): Motion_{average} = 0.8965
    - Good example (Right): Motion_{average} = 0.9028

    <table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
    <tr>
        <td>
            <video src="https://github.com/user-attachments/assets/b0bc3083-3b28-4cc8-8944-364b4ef617e8" width="100%" controls loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/5a60f6be-0d49-4e8b-ad9a-c067c13ebd14" width="100%" controls loop></video>
        </td>
    </tr>
    </table>
