<p align="center"><img src="assets/logo.jpg" width="25%"></p>
<div align="center">
  <a href='https://arxiv.org/abs/2512.02793'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://github.com/wufan-cse/IC-World'><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp;
  </a>
</div>

This is the official implementation for [paper](https://arxiv.org/abs/2512.02793), IC-World: In-Context Generation for Shared World Modeling.

<!-- ## Key Features

IC-World has the following features:
- Support GRPO finetuning + Wan-2.1-I2V-14B -->

## Updates

- __[2025.12.13]__: We released evaluation codes in [benchmark](https://github.com/wufan-cse/IC-World/tree/main/benchmark).
- __[2025.12.03]__: 🔥 We released the paper in arXiv!

If you have any research or engineering inquiries, feel free to open issues or email us directly at fan011@e.ntu.edu.sg.

## TODOs

- [ ] Support more video foundation models.
- [ ] Release training codes (before 2026.02.28).
- [ ] Release dataset (before 2025.12.31).
- [ ] Release inference codes (before 2025.12.31).
- [x] Release evaluation metrics codes.
- [x] Release paper.

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


## Video Demos of IC-World

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


## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) & [Wan2.2](https://github.com/Wan-Video/Wan2.2)
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
