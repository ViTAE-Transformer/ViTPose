<h1 align="left">ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation<a href="https://arxiv.org/abs/2204.12484"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitpose-simple-vision-transformer-baselines/pose-estimation-on-coco-test-dev)](https://paperswithcode.com/sota/pose-estimation-on-coco-test-dev?p=vitpose-simple-vision-transformer-baselines)

<p align="center">
  <a href="#Results">Results</a> |
  <a href="#Updates">Updates</a> |
  <a href="#Usage">Usage</a> |
  <a href='#Todo'>Todo</a> |
  <a href="#Acknowledge">Acknowledge</a>
</p>

<p align="center">
<a href="https://giphy.com/gifs/UfPQB1qKir7Vqem6sL/fullscreen"><img src="https://media.giphy.com/media/ZewXwZuixYKS2lZmNL/giphy.gif"></a>   <a href="https://giphy.com/gifs/DCvf1DrWZgbwPa8bWZ/fullscreen"><img src="https://media.giphy.com/media/2AEeuicbIjwqp2mbug/giphy.gif"></a>
</p>
<p align="center">
<a href="https://giphy.com/gifs/r3GaZz7H1H6zpuIvPI/fullscreen"><img src="https://media.giphy.com/media/13oe6zo6b2B7CdsOac/giphy.gif"></a>    <a href="https://giphy.com/gifs/FjzrGJxsOzZAXaW7Vi/fullscreen"><img src="https://media.giphy.com/media/4JLERHxOEgH0tt5DZO/giphy.gif"></a>
</p>

This branch contains the pytorch implementation of <a href="https://arxiv.org/abs/2204.12484">ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation</a>. It obtains 81.1 AP on MS COCO Keypoint test-dev set.

## Results from this repo on MS COCO val set

Using detection results from a detector that obtains 56 mAP on person.

| Model | Pretrain | Resolution | AP | AR | config | log | weight |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| ViT-Base | MAE | 256x192 | 75.8 | 81.1 | config | log |  |
| ViT-Large | MAE | 256x192 | 78.3 | 83.5 | config | log |  |
| ViT-Huge | MAE | 256x192 | 79.1 | 84.1 | config | log |  |

## Updates

> [2022-04-27] Our ViTPose with ViTAE-G obtains 81.1 AP on COCO test-dev set! 

> Applications of ViTAE Transformer include: [image classification](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Image-Classification) | [object detection](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Object-Detection) | [semantic segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Semantic-Segmentation) | [animal pose segmentation](https://github.com/ViTAE-Transformer/ViTAE-Transformer/tree/main/Animal-Pose-Estimation) | [remote sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) | [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) | [ViTDet](https://github.com/ViTAE-Transformer/ViTDet)

## Usage

We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

Download the pretrained models from [MAE](https://github.com/facebookresearch/mae) or [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer), and then conduct the experiments by

```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch --seed 0
```

## Todo

This repo current contains modifications including:

- [ ] Upload configs and pretrained models

- [ ] More models with SOTA results

## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection) and [MAE](https://github.com/facebookresearch/mae).

## Citing ViTPose
```
@misc{xu2022vitpose,
      title={ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation}, 
      author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
      year={2022},
      eprint={2204.12484},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For ViTAE and ViTAEv2, please refer to:
```
@article{xu2021vitae,
  title={Vitae: Vision transformer advanced by exploring intrinsic inductive bias},
  author={Xu, Yufei and Zhang, Qiming and Zhang, Jing and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{zhang2022vitaev2,
  title={ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond},
  author={Zhang, Qiming and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2202.10108},
  year={2022}
}
```
