<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html">DeepPose (CVPR'2014)</a></summary>

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

</details>

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
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
| [deeppose_resnet_50](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py) |  256x192   | 0.541 |      0.824      |      0.601      | 0.649 |      0.893      | [ckpt](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.pth) | [log](https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.log.json) |
| [deeppose_resnet_101](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res101_8xb64-210e_coco-256x192.py) |  256x192   | 0.562 |      0.831      |      0.629      | 0.670 |      0.900      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192_20210205.log.json) |
| [deeppose_resnet_152](/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_8xb64-210e_coco-256x192.py) |  256x192   | 0.584 |      0.842      |      0.659      | 0.688 |      0.907      | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192_20210205.log.json) |
