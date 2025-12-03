<p align="center"><img src="assets/logo.jpg" width="30%"></p>
<div align="center">
  <a href='https://arxiv.org/abs/2512.02793'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://github.com/wufan-cse/IC-World'><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp;
  </a>&nbsp;
</div>

This is the official implementation for [paper](https://arxiv.org/abs/2512.02793), IC-World: In-Context Generation for Shared World Modeling.

## Key Features

DanceGRPO has the following features:
- Support GRPO finetuning
- Support Wan-2.1-I2V-14B

## Updates

- __[2025.12.03]__: 🔥 We released the paper in arXiv!

If you have any research or engineering inquiries, feel free to open issues or email us directly at fan011@e.ntu.edu.sg.

## TODOs

- [ ] Support more video foundation models.
- [ ] Release training codes (before 2026.02.28).
- [ ] Release evaluation metrics codes (before 2025.12.31).
- [ ] Release dataset (before 2025.12.31).
- [ ] Release inference codes (before 2025.12.31).
- [x] Release paper.

## Video Demos

<!-- Here we present some rough demos for quick look. -->

#### Static scene + Dynamic camera

IC-World presents strong capability in maintaining shared world consistency, ensuring that different viewpoints observe the same underlying environment. The following demos highlight this behavior: scenes and objects reappear across views. In the left example, the letter above the door first appears in the left view and later reappears in the right view. In the right example, the advertising tag on the lower-right table first appears in the right view and subsequently reappears in the left view.

<video src="https://github.com/wufan-cse/IC-World/blob/main/assets/demo1.mp4" width="48%" controls loop></video>
<video src="https://github.com/wufan-cse/IC-World/blob/main/assets/demo2.mp4" width="48%" controls loop></video>

#### Dynamic scene + Static camera

<video src="https://github.com/wufan-cse/IC-World/blob/main/assets/demo3.mp4" width="48%" controls loop></video>


## Acknowledgement
We learned and reused code from the following projects:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)

We thank the authors for their contributions to the community!

## Citation
If you find IC-World useful for your research, please cite our paper:

```bibtex
@article{wu2025icworld,
  title={IC-World: In-Context Generation for Shared World Modeling},
  author={Wu, Fan and Wei, Jiacheng and Li, Ruibo and Xu, Yi and Li, Junyou and Ye, Deheng and Lin, Guosheng},
  journal={arXiv preprint arXiv:2512.02793},
  year={2025}
}
```
