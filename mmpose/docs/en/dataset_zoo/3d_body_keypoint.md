# 3D Body Keypoint Datasets

It is recommended to symlink the dataset root to `$MMPOSE/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

MMPose supported datasets:

- [Human3.6M](#human36m) \[ [Homepage](http://vision.imar.ro/human3.6m/description.php) \]
- [CMU Panoptic](#cmu-panoptic) \[ [Homepage](http://domedb.perception.cs.cmu.edu/) \]
- [Campus/Shelf](#campus-and-shelf) \[ [Homepage](http://campar.in.tum.de/Chair/MultiHumanPose) \]
- [UBody](#ubody3d) \[ [Homepage](https://osx-ubody.github.io/) \]

## Human3.6M

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  volume = {36},
  number = {7},
  pages = {1325-1339},
  month = {jul},
  year = {2014}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227866174-0355e7dd-6965-4d1a-8bb6-32e8dce4cefc.png" height="300px">
</div>

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php), please download from the official website and run the [preprocessing script](/tools/dataset_converters/preprocess_h36m.py), which will extract camera parameters and pose annotations at full framerate (50 FPS) and downsampled framerate (10 FPS). The processed data should have the following structure:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── h36m
        ├── annotation_body3d
        |   ├── cameras.pkl
        |   ├── fps50
        |   |   ├── h36m_test.npz
        |   |   ├── h36m_train.npz
        |   |   ├── joint2d_rel_stats.pkl
        |   |   ├── joint2d_stats.pkl
        |   |   ├── joint3d_rel_stats.pkl
        |   |   `── joint3d_stats.pkl
        |   `── fps10
        |       ├── h36m_test.npz
        |       ├── h36m_train.npz
        |       ├── joint2d_rel_stats.pkl
        |       ├── joint2d_stats.pkl
        |       ├── joint3d_rel_stats.pkl
        |       `── joint3d_stats.pkl
        `── images
            ├── S1
            |   ├── S1_Directions_1.54138969
            |   |   ├── S1_Directions_1.54138969_00001.jpg
            |   |   ├── S1_Directions_1.54138969_00002.jpg
            |   |   ├── ...
            |   ├── ...
            ├── S5
            ├── S6
            ├── S7
            ├── S8
            ├── S9
            `── S11
```

## CMU Panoptic

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_iccv_2015/html/Joo_Panoptic_Studio_A_ICCV_2015_paper.html">CMU Panoptic (ICCV'2015)</a></summary>

```bibtex
@Article = {joo_iccv_2015,
author = {Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh},
title = {Panoptic Studio: A Massively Multiview System for Social Motion Capture},
booktitle = {ICCV},
year = {2015}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227866246-1af9441b-4f6c-4340-ae98-e40d6b4e8e55.png" height="300px">
</div>

Please follow [voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch) to prepare this dataset.

1. Download the dataset by following the instructions in [panoptic-toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) and extract them under `$MMPOSE/data/panoptic`.

2. Only download those sequences that are needed. You can also just download a subset of camera views by specifying the number of views (HD_Video_Number) and changing the camera order in `./scripts/getData.sh`. The used sequences and camera views can be found in [VoxelPose](https://arxiv.org/abs/2004.06239). Note that the sequence "160906_band3" might not be available due to errors on the server of CMU Panoptic.

3. Note that we only use HD videos,  calibration data, and 3D Body Keypoint in the codes. You can comment out other irrelevant codes such as downloading 3D Face data in `./scripts/getData.sh`.

The directory tree should be like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── panoptic
        ├── 16060224_haggling1
        |   |   ├── hdImgs
        |   |   ├── hdvideos
        |   |   ├── hdPose3d_stage1_coco19
        |   |   ├── calibration_160224_haggling1.json
        ├── 160226_haggling1
            ├── ...
```

## Campus and Shelf

<details>
<summary align="right"><a href="http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf">Campus and Shelf (CVPR'2014)</a></summary>

```bibtex
@inproceedings {belagian14multi,
    title = {{3D} Pictorial Structures for Multiple Human Pose Estimation},
    author = {Belagiannis, Vasileios and Amin, Sikandar and Andriluka, Mykhaylo and Schiele, Bernt and Navab
    Nassir and Ilic, Slobo
    booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June},
    organization={IEEE}
}
```

</details>

<div align="center">
  <img src="https://user-images.githubusercontent.com/100993824/227866373-ed8e6d1a-581b-4b88-93f5-2bc2caddbf6c.png" height="300px">
</div>

Please follow [voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch) to prepare these two datasets.

1. Please download the datasets from the [official website](http://campar.in.tum.de/Chair/MultiHumanPose) and extract them under `$MMPOSE/data/campus` and `$MMPOSE/data/shelf`, respectively. The original data include images as well as the ground truth pose file `actorsGT.mat`.

2. We directly use the processed camera parameters from [voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch). You can download them from this repository and place in under `$MMPOSE/data/campus/calibration_campus.json` and `$MMPOSE/data/shelf/calibration_shelf.json`, respectively.

3. Like [Voxelpose](https://github.com/microsoft/voxelpose-pytorch), due to the limited and incomplete annotations of the two datasets, we don't train the model using this dataset. Instead, we directly use the 2D pose estimator trained on COCO, and use independent 3D human poses from the CMU Panoptic dataset to train our 3D model. It lies in `${MMPOSE}/data/panoptic_training_pose.pkl`.

4. Like [Voxelpose](https://github.com/microsoft/voxelpose-pytorch), for testing, we first estimate 2D poses and generate 2D heatmaps for these two datasets. You can download the predicted poses from [voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch) and place them in  `$MMPOSE/data/campus/pred_campus_maskrcnn_hrnet_coco.pkl` and `$MMPOSE/data/shelf/pred_shelf_maskrcnn_hrnet_coco.pkl`, respectively. You can also use the models trained on COCO dataset (like HigherHRNet) to generate 2D heatmaps directly.

The directory tree should be like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    ├── panoptic_training_pose.pkl
    ├── campus
    |   ├── Camera0
    |   |   |   ├── campus4-c0-00000.png
    |   |   |   ├── ...
    |   |   |   ├── campus4-c0-01999.png
    |   ...
    |   ├── Camera2
    |   |   |   ├── campus4-c2-00000.png
    |   |   |   ├── ...
    |   |   |   ├── campus4-c2-01999.png
    |   ├── calibration_campus.json
    |   ├── pred_campus_maskrcnn_hrnet_coco.pkl
    |   ├── actorsGT.mat
    ├── shelf
    |   ├── Camera0
    |   |   |   ├── img_000000.png
    |   |   |   ├── ...
    |   |   |   ├── img_003199.png
    |   ...
    |   ├── Camera4
    |   |   |   ├── img_000000.png
    |   |   |   ├── ...
    |   |   |   ├── img_003199.png
    |   ├── calibration_shelf.json
    |   ├── pred_shelf_maskrcnn_hrnet_coco.pkl
    |   ├── actorsGT.mat
```

## UBody3d

<details>
<summary align="right"><a href="https://arxiv.org/abs/2303.16160">UBody (CVPR'2023)</a></summary>

```bibtex
@article{lin2023one,
  title={One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer},
  author={Lin, Jing and Zeng, Ailing and Wang, Haoqian and Zhang, Lei and Li, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023},
}
```

</details>

<div align="center">
  <img src="https://github.com/open-mmlab/mmpose/assets/15952744/0c97e43a-46a9-46a3-a5dd-b84bf9d6d6f2" height="300px">
</div>

For [Ubody](https://github.com/IDEA-Research/OSX) dataset, videos and annotations can be downloaded from [OSX homepage](https://github.com/IDEA-Research/OSX).

Download and extract them under $MMPOSE/data, and make them look like this:

```text
mmpose
├── mmpose
├── docs
├── tests
├── tools
├── configs
`── data
    │── UBody
        ├── annotations
        │   ├── ConductMusic
        │   ├── Entertainment
        │   ├── Fitness
        │   ├── Interview
        │   ├── LiveVlog
        │   ├── Magic_show
        │   ├── Movie
        │   ├── Olympic
        │   ├── Online_class
        │   ├── SignLanguage
        │   ├── Singing
        │   ├── Speech
        │   ├── TVShow
        │   ├── TalkShow
        │   └── VideoConference
        ├── splits
        │   ├── inter_scene_test_list.npy
        │   └── intra_scene_test_list.npy
        ├── videos
        │   ├── ConductMusic
        │   ├── Entertainment
        │   ├── Fitness
        │   ├── Interview
        │   ├── LiveVlog
        │   ├── Magic_show
        │   ├── Movie
        │   ├── Olympic
        │   ├── Online_class
        │   ├── SignLanguage
        │   ├── Singing
        │   ├── Speech
        │   ├── TVShow
        │   ├── TalkShow
        │   └── VideoConference
```

Convert videos to images then split them into train/val set:

```shell
python tools/dataset_converters/ubody_kpts_to_coco.py
```

Before generating 3D keypoints, you need to install SMPLX tools and download human models, please refer to [Github](https://github.com/vchoutas/smplx#installation) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php).

```shell
pip install smplx
```

The directory tree of human models should be like this:

```text
human_model_path
|── smplx
    ├── SMPLX_NEUTRAL.npz
    ├── SMPLX_NEUTRAL.pkl
```

After the above preparations are finished, execute the following script:

```shell
python tools/dataset_converters/ubody_smplx_to_coco.py --data-root {$MMPOSE/data/UBody} --human-model-path {$MMPOSE/data/human_model_path/}
```
