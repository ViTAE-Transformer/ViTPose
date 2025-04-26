# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .resnet import BasicBlock, ResLayer


class HourglassModule(BaseModule):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 init_cfg=None):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_block, cur_channel, cur_channel, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_block,
            cur_channel,
            next_channel,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_block,
                next_channel,
                next_channel,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            cur_block,
            next_channel,
            cur_channel,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@MODELS.register_module()
class HourglassNet(BaseBackbone):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default:
            ``[
                dict(type='Normal', std=0.001, layer=['Conv2d']),
                dict(
                    type='Constant',
                    val=1,
                    layer=['_BatchNorm', 'GroupNorm'])
            ]``

    Example:
        >>> from mmpose.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(
        self,
        downsample_times=5,
        num_stacks=2,
        stage_channels=(256, 256, 384, 384, 384, 512),
        stage_blocks=(2, 2, 2, 2, 2, 4),
        feat_channel=256,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
    ):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 1, 128, 256, stride=2, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            num_stacks - 1,
            cur_channel,
            cur_channel,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))

        return out_feats
