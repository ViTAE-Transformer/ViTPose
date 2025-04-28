# --------------------------------------------------------
# Swin Transformer MoE
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from tutel import moe as tutel_moe
except ImportError:
    tutel_moe = None
    print(
        'Tutel has not been installed. To use Swin-MoE, please install Tutel;'
        'otherwise, just ignore this.')


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 mlp_fc2_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_fc2_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 num_local_experts,
                 top_value,
                 capacity_factor=1.25,
                 cosine_router=False,
                 normalize_gate=False,
                 use_bpr=True,
                 is_gshard_loss=True,
                 gate_noise=1.0,
                 cosine_router_dim=256,
                 cosine_router_init_t=0.5,
                 moe_drop=0.0,
                 init_std=0.02,
                 mlp_fc2_bias=True):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_local_experts = num_local_experts
        self.top_value = top_value
        self.capacity_factor = capacity_factor
        self.cosine_router = cosine_router
        self.normalize_gate = normalize_gate
        self.use_bpr = use_bpr
        self.init_std = init_std
        self.mlp_fc2_bias = mlp_fc2_bias

        self.dist_rank = dist.get_rank()

        self._dropout = nn.Dropout(p=moe_drop)

        _gate_type = {
            'type': 'cosine_top' if cosine_router else 'top',
            'k': top_value,
            'capacity_factor': capacity_factor,
            'gate_noise': gate_noise,
            'fp32_gate': True
        }
        if cosine_router:
            _gate_type['proj_dim'] = cosine_router_dim
            _gate_type['init_t'] = cosine_router_init_t
        self._moe_layer = tutel_moe.moe_layer(
            gate_type=_gate_type,
            model_dim=in_features,
            experts={
                'type': 'ffn',
                'count_per_node': num_local_experts,
                'hidden_size_per_expert': hidden_features,
                'activation_fn': lambda x: self._dropout(F.gelu(x))
            },
            scan_expert_func=lambda name, param: setattr(
                param, 'skip_allreduce', True),
            seeds=(1, self.dist_rank + 1, self.dist_rank + 1),
            batch_prioritized_routing=use_bpr,
            normalize_gate=normalize_gate,
            is_gshard_loss=is_gshard_loss,
        )
        if not self.mlp_fc2_bias:
            self._moe_layer.experts.batched_fc2_bias.requires_grad = False

    def forward(self, x):
        x = self._moe_layer(x)
        return x, x.l_aux

    def extra_repr(self) -> str:
        return (f'[Statistics-{self.dist_rank}] param count for MoE, '
                f'in_features = {self.in_features}, '
                f'hidden_features = {self.hidden_features}, '
                f'num_local_experts = {self.num_local_experts}, '
                f'top_value = {self.top_value}, '
                f'cosine_router={self.cosine_router} '
                f'normalize_gate={self.normalize_gate}, '
                f'use_bpr = {self.use_bpr}')

    def _init_weights(self):
        if hasattr(self._moe_layer, 'experts'):
            trunc_normal_(
                self._moe_layer.experts.batched_fc1_w, std=self.init_std)
            trunc_normal_(
                self._moe_layer.experts.batched_fc2_w, std=self.init_std)
            nn.init.constant_(self._moe_layer.experts.batched_fc1_bias, 0)
            nn.init.constant_(self._moe_layer.experts.batched_fc2_bias, 0)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative
    position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query,
        key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
        head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
        Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the
        window in pretraining.
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1),
            self.window_size[0],
            dtype=torch.float32)
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1),
            self.window_size[1],
            dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])).permute(
                1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (
                pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (
                pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer('relative_coords_table', relative_coords_table)

        # get pair-wise relative position index for each token inside the
        # window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index',
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or
            None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
            (f'pretrained_window_size={self.pretrained_window_size}, '
             f'num_heads={self.num_heads}')

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query,
        key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
        head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default:
        nn.LayerNorm
        mlp_fc2_bias (bool): Whether to add bias in fc2 of Mlp. Default: True
        init_std: Initialization std. Default: 0.02
        pretrained_window_size (int): Window size in pretraining.
        is_moe (bool): If True, this block is a MoE block.
        num_local_experts (int): number of local experts in each device (
        GPU). Default: 1
        top_value (int): the value of k in top-k gating. Default: 1
        capacity_factor (float): the capacity factor in MoE. Default: 1.25
        cosine_router (bool): Whether to use cosine router. Default: False
        normalize_gate (bool): Whether to normalize the gating score in top-k
        gating. Default: False
        use_bpr (bool): Whether to use batch-prioritized-routing. Default: True
        is_gshard_loss (bool): If True, use Gshard balance loss.
                               If False, use the load loss and importance
                               loss in "arXiv:1701.06538". Default: False
        gate_noise (float): the noise ratio in top-k gating. Default: 1.0
        cosine_router_dim (int): Projection dimension in cosine router.
        cosine_router_init_t (float): Initialization temperature in cosine
        router.
        moe_drop (float): Dropout rate in MoE. Default: 0.0
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mlp_fc2_bias=True,
                 init_std=0.02,
                 pretrained_window_size=0,
                 is_moe=False,
                 num_local_experts=1,
                 top_value=1,
                 capacity_factor=1.25,
                 cosine_router=False,
                 normalize_gate=False,
                 use_bpr=True,
                 is_gshard_loss=True,
                 gate_noise=1.0,
                 cosine_router_dim=256,
                 cosine_router_init_t=0.5,
                 moe_drop=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_moe = is_moe
        self.capacity_factor = capacity_factor
        self.top_value = top_value

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't
            # partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, ('shift_size must in '
                                                         '0-window_size')

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.is_moe:
            self.mlp = MoEMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_drop=moe_drop,
                mlp_fc2_bias=mlp_fc2_bias,
                init_std=init_std)
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                mlp_fc2_bias=mlp_fc2_bias)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size,
                                   C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H,
                                   W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        shortcut = x
        x = self.norm2(x)
        if self.is_moe:
            x, l_aux = self.mlp(x)
            x = shortcut + self.drop_path(x)
            return x, l_aux
        else:
            x = shortcut + self.drop_path(self.mlp(x))
            return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, '
                f'input_resolution={self.input_resolution}, '
                f'num_heads={self.num_heads}, '
                f'window_size={self.window_size}, '
                f'shift_size={self.shift_size}, '
                f'mlp_ratio={self.mlp_ratio}')

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        if self.is_moe:
            flops += (2 * H * W * self.dim * self.dim * self.mlp_ratio *
                      self.capacity_factor * self.top_value)
        else:
            flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default:
        nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query,
        key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
        head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate.
        Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default:
        nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end
        of the layer. Default: None
        mlp_fc2_bias (bool): Whether to add bias in fc2 of Mlp. Default: True
        init_std: Initialization std. Default: 0.02
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        Default: False.
        pretrained_window_size (int): Local window size in pretraining.
        moe_blocks (tuple(int)): The index of each MoE block.
        num_local_experts (int): number of local experts in each device (
        GPU). Default: 1
        top_value (int): the value of k in top-k gating. Default: 1
        capacity_factor (float): the capacity factor in MoE. Default: 1.25
        cosine_router (bool): Whether to use cosine router Default: False
        normalize_gate (bool): Whether to normalize the gating score in top-k
        gating. Default: False
        use_bpr (bool): Whether to use batch-prioritized-routing. Default: True
        is_gshard_loss (bool): If True, use Gshard balance loss.
                               If False, use the load loss and importance
                               loss in "arXiv:1701.06538". Default: False
        gate_noise (float): the noise ratio in top-k gating. Default: 1.0
        cosine_router_dim (int): Projection dimension in cosine router.
        cosine_router_init_t (float): Initialization temperature in cosine
        router.
        moe_drop (float): Dropout rate in MoE. Default: 0.0
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 mlp_fc2_bias=True,
                 init_std=0.02,
                 use_checkpoint=False,
                 pretrained_window_size=0,
                 moe_block=[-1],
                 num_local_experts=1,
                 top_value=1,
                 capacity_factor=1.25,
                 cosine_router=False,
                 normalize_gate=False,
                 use_bpr=True,
                 is_gshard_loss=True,
                 cosine_router_dim=256,
                 cosine_router_init_t=0.5,
                 gate_noise=1.0,
                 moe_drop=0.0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                mlp_fc2_bias=mlp_fc2_bias,
                init_std=init_std,
                pretrained_window_size=pretrained_window_size,
                is_moe=True if i in moe_block else False,
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_drop=moe_drop) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        l_aux = 0.0
        for blk in self.blocks:
            if self.use_checkpoint:
                out = checkpoint.checkpoint(blk, x)
            else:
                out = blk(x)
            if isinstance(out, tuple):
                x = out[0]
                cur_l_aux = out[1]
                l_aux = cur_l_aux + l_aux
            else:
                x = out

        if self.downsample is not None:
            x = self.downsample(x)
        return x, l_aux

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, '
                f'depth={self.depth}')

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels.
        Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            (f"Input image size ({H}*{W}) doesn't match model ("
             f'{self.img_size[0]}*{self.img_size[1]}).')
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerMoE(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision
        Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head.
        Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if
        set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch
        embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding.
        Default: True
        mlp_fc2_bias (bool): Whether to add bias in fc2 of Mlp. Default: True
        init_std: Initialization std. Default: 0.02
        use_checkpoint (bool): Whether to use checkpointing to save memory.
        Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each
        layer.
        moe_blocks (tuple(tuple(int))): The index of each MoE block in each
        layer.
        num_local_experts (int): number of local experts in each device (
        GPU). Default: 1
        top_value (int): the value of k in top-k gating. Default: 1
        capacity_factor (float): the capacity factor in MoE. Default: 1.25
        cosine_router (bool): Whether to use cosine router Default: False
        normalize_gate (bool): Whether to normalize the gating score in top-k
        gating. Default: False
        use_bpr (bool): Whether to use batch-prioritized-routing. Default: True
        is_gshard_loss (bool): If True, use Gshard balance loss.
                               If False, use the load loss and importance
                               loss in "arXiv:1701.06538". Default: False
        gate_noise (float): the noise ratio in top-k gating. Default: 1.0
        cosine_router_dim (int): Projection dimension in cosine router.
        cosine_router_init_t (float): Initialization temperature in cosine
        router.
        moe_drop (float): Dropout rate in MoE. Default: 0.0
        aux_loss_weight (float): auxiliary loss weight. Default: 0.1
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 mlp_fc2_bias=True,
                 init_std=0.02,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 moe_blocks=[[-1], [-1], [-1], [-1]],
                 num_local_experts=1,
                 top_value=1,
                 capacity_factor=1.25,
                 cosine_router=False,
                 normalize_gate=False,
                 use_bpr=True,
                 is_gshard_loss=True,
                 gate_noise=1.0,
                 cosine_router_dim=256,
                 cosine_router_init_t=0.5,
                 moe_drop=0.0,
                 aux_loss_weight=0.01,
                 **kwargs):
        super().__init__()
        self._ddp_params_and_buffers_to_ignore = list()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.init_std = init_std
        self.aux_loss_weight = aux_loss_weight
        self.num_local_experts = num_local_experts
        self.global_experts = num_local_experts * dist.get_world_size() if (
                num_local_experts > 0) \
            else dist.get_world_size() // (-num_local_experts)
        self.sharded_count = (
            1.0 / num_local_experts) if num_local_experts > 0 else (
                -num_local_experts)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=self.init_std)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if
                (i_layer < self.num_layers - 1) else None,
                mlp_fc2_bias=mlp_fc2_bias,
                init_std=init_std,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                moe_block=moe_blocks[i_layer],
                num_local_experts=num_local_experts,
                top_value=top_value,
                capacity_factor=capacity_factor,
                cosine_router=cosine_router,
                normalize_gate=normalize_gate,
                use_bpr=use_bpr,
                is_gshard_loss=is_gshard_loss,
                gate_noise=gate_noise,
                cosine_router_dim=cosine_router_dim,
                cosine_router_init_t=cosine_router_init_t,
                moe_drop=moe_drop)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features,
            num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, MoEMlp):
            m._init_weights()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {
            'cpb_mlp', 'relative_position_bias_table', 'fc1_bias', 'fc2_bias',
            'temperature', 'cosine_projector', 'sim_matrix'
        }

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        l_aux = 0.0
        for layer in self.layers:
            x, cur_l_aux = layer(x)
            l_aux = cur_l_aux + l_aux

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, l_aux

    def forward(self, x):
        x, l_aux = self.forward_features(x)
        x = self.head(x)
        return x, l_aux * self.aux_loss_weight

    def add_param_to_skip_allreduce(self, param_name):
        self._ddp_params_and_buffers_to_ignore.append(param_name)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
