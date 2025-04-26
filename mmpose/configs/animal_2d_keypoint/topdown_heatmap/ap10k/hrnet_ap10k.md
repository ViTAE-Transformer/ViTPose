<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2108.12617">AP-10K (NeurIPS'2021)</a></summary>

```bibtex
@misc{yu2021ap10k,
      title={AP-10K: A Benchmark for Animal Pose Estimation in the Wild},
      author={Hang Yu and Yufei Xu and Jing Zhang and Wei Zhao and Ziyu Guan and Dacheng Tao},
      year={2021},
      eprint={2108.12617},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</details>

Results on AP-10K validation set

| Arch                                       | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> | AP<sup>M</sup> | AP<sup>L</sup> |                    ckpt                     |                    log                     |
| :----------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :------------: | :------------: | :-----------------------------------------: | :----------------------------------------: |
| [pose_hrnet_w32](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w32_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.722 |      0.935      |      0.789      |     0.557      |     0.729      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_ap10k_256x256-18aac840_20211029.log.json) |
| [pose_hrnet_w48](/configs/animal_2d_keypoint/topdown_heatmap/ap10k/td-hm_hrnet-w48_8xb64-210e_ap10k-256x256.py) |  256x256   | 0.728 |      0.936      |      0.802      |     0.577      |     0.735      | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_ap10k_256x256-d95ab412_20211029.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_ap10k_256x256-d95ab412_20211029.log.json) |
