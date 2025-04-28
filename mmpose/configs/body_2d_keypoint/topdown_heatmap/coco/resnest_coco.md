<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/2004.08955">ResNeSt (ArXiv'2020)</a></summary>

```bibtex
@article{zhang2020resnest,
  title={ResNeSt: Split-Attention Networks},
  author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Zhang, Zhi and Lin, Haibin and Sun, Yue and He, Tong and Muller, Jonas and Manmatha, R. and Li, Mu and Smola, Alexander},
  journal={arXiv preprint arXiv:2004.08955},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48">COCO (ECCV'2014)</a></summary>

```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft coco: Common objects in context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  booktitle={European conference on computer vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

</details>

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch                                          | Input Size |  AP   | AP<sup>50</sup> | AP<sup>75</sup> |  AR   | AR<sup>50</sup> |                     ckpt                      |                      log                      |
| :-------------------------------------------- | :--------: | :---: | :-------------: | :-------------: | :---: | :-------------: | :-------------------------------------------: | :-------------------------------------------: |
| [pose_resnest_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest50_8xb64-210e_coco-256x192.py) |  256x192   | 0.720 |      0.899      |      0.800      | 0.775 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192-6e65eece_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192_20210320.log.json) |
| [pose_resnest_50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest50_8xb64-210e_coco-384x288.py) |  384x288   | 0.737 |      0.900      |      0.811      | 0.789 |      0.937      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288-dcd20436_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288_20210320.log.json) |
| [pose_resnest_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest101_8xb64-210e_coco-256x192.py) |  256x192   | 0.725 |      0.900      |      0.807      | 0.781 |      0.939      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192-2ffcdc9d_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192_20210320.log.json) |
| [pose_resnest_101](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest101_8xb32-210e_coco-384x288.py) |  384x288   | 0.745 |      0.905      |      0.818      | 0.798 |      0.942      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288-80660658_20210320.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288_20210320.log.json) |
| [pose_resnest_200](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest200_8xb64-210e_coco-256x192.py) |  256x192   | 0.731 |      0.905      |      0.812      | 0.787 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_256x192-db007a48_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_256x192_20210517.log.json) |
| [pose_resnest_200](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest200_8xb16-210e_coco-384x288.py) |  384x288   | 0.753 |      0.907      |      0.827      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_384x288-b5bb76cb_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_384x288_20210517.log.json) |
| [pose_resnest_269](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest269_8xb32-210e_coco-256x192.py) |  256x192   | 0.737 |      0.907      |      0.819      | 0.792 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_256x192-2a7882ac_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_256x192_20210517.log.json) |
| [pose_resnest_269](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_resnest269_8xb16-210e_coco-384x288.py) |  384x288   | 0.754 |      0.908      |      0.828      | 0.805 |      0.943      | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_384x288-b142b9fb_20210517.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_384x288_20210517.log.json) |
