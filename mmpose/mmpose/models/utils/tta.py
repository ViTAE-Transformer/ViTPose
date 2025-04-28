# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

import warnings
import numpy as np
import cv2


def flip_back(output_flipped, flip_pairs, target_type="GaussianHeatmap"):
    """Flip the flipped heatmaps back to the original form.

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output_flipped (np.ndarray[N, K, H, W]): The output heatmaps obtained
            from the flipped images.
        flip_pairs (list[tuple()): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).
        target_type (str): GaussianHeatmap or CombinedTarget

    Returns:
        np.ndarray: heatmaps that flipped back to the original image
    """
    assert output_flipped.ndim == 4, "output_flipped should be [batch_size, num_keypoints, height, width]"
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == "CombinedTarget".lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels, shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    return output_flipped_back


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def flip_heatmaps(
    heatmaps: Tensor, flip_indices: Optional[List[int]] = None, flip_mode: str = "heatmap", shift_heatmap: bool = True
):
    """Flip heatmaps for test-time augmentation.

    Args:
        heatmaps (Tensor): The heatmaps to flip. Should be a tensor in shape
            [B, C, H, W]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint. Defaults to ``None``
        flip_mode (str): Specify the flipping mode. Options are:

            - ``'heatmap'``: horizontally flip the heatmaps and swap heatmaps
                of symmetric keypoints according to ``flip_indices``
            - ``'udp_combined'``: similar to ``'heatmap'`` mode but further
                flip the x_offset values
            - ``'offset'``: horizontally flip the offset fields and swap
                heatmaps of symmetric keypoints according to
                ``flip_indices``. x_offset values are also reversed
        shift_heatmap (bool): Shift the flipped heatmaps to align with the
            original heatmaps and improve accuracy. Defaults to ``True``

    Returns:
        Tensor: flipped heatmaps in shape [B, C, H, W]
    """

    if flip_mode == "heatmap":
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == heatmaps.shape[1]
            heatmaps = heatmaps[:, flip_indices]
    elif flip_mode == "udp_combined":
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C // 3, 3, H, W)
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == C // 3
            heatmaps = heatmaps[:, flip_indices]
        heatmaps[:, :, 1] = -heatmaps[:, :, 1]
        heatmaps = heatmaps.view(B, C, H, W)

    elif flip_mode == "offset":
        B, C, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, C // 2, -1, H, W)
        heatmaps = heatmaps.flip(-1)
        if flip_indices is not None:
            assert len(flip_indices) == C // 2
            heatmaps = heatmaps[:, flip_indices]
        heatmaps[:, :, 0] = -heatmaps[:, :, 0]
        heatmaps = heatmaps.view(B, C, H, W)

    else:
        raise ValueError(f'Invalid flip_mode value "{flip_mode}"')

    if shift_heatmap:
        # clone data to avoid unexpected in-place operation when using CPU
        heatmaps[..., 1:] = heatmaps[..., :-1].clone()

    return heatmaps


def flip_vectors(x_labels: Tensor, y_labels: Tensor, flip_indices: List[int]):
    """Flip instance-level labels in specific axis for test-time augmentation.

    Args:
        x_labels (Tensor): The vector labels in x-axis to flip. Should be
            a tensor in shape [B, C, Wx]
        y_labels (Tensor): The vector labels in y-axis to flip. Should be
            a tensor in shape [B, C, Wy]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
    """
    assert x_labels.ndim == 3 and y_labels.ndim == 3
    assert len(flip_indices) == x_labels.shape[1] and len(flip_indices) == y_labels.shape[1]
    x_labels = x_labels[:, flip_indices].flip(-1)
    y_labels = y_labels[:, flip_indices]

    return x_labels, y_labels


def flip_coordinates(coords: Tensor, flip_indices: List[int], shift_coords: bool, input_size: Tuple[int, int]):
    """Flip normalized coordinates for test-time augmentation.

    Args:
        coords (Tensor): The coordinates to flip. Should be a tensor in shape
            [B, K, D]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
        shift_coords (bool): Shift the flipped coordinates to align with the
            original coordinates and improve accuracy. Defaults to ``True``
        input_size (Tuple[int, int]): The size of input image in [w, h]
    """
    assert coords.ndim == 3
    assert len(flip_indices) == coords.shape[1]

    coords[:, :, 0] = 1.0 - coords[:, :, 0]

    if shift_coords:
        img_width = input_size[0]
        coords[:, :, 0] -= 1.0 / img_width

    coords = coords[:, flip_indices]
    return coords


def flip_visibility(vis: Tensor, flip_indices: List[int]):
    """Flip keypoints visibility for test-time augmentation.

    Args:
        vis (Tensor): The keypoints visibility to flip. Should be a tensor
            in shape [B, K]
        flip_indices (List[int]): The indices of each keypoint's symmetric
            keypoint
    """
    assert vis.ndim == 2

    vis = vis[:, flip_indices]
    return vis


def aggregate_heatmaps(
    heatmaps: List[Tensor], size: Optional[Tuple[int, int]], align_corners: bool = False, mode: str = "average"
):
    """Aggregate multiple heatmaps.

    Args:
        heatmaps (List[Tensor]): Multiple heatmaps to aggregate. Each should
            be in shape (B, C, H, W)
        size (Tuple[int, int], optional): The target size in (w, h). All
            heatmaps will be resized to the target size. If not given, the
            first heatmap tensor's width and height will be used as the target
            size. Defaults to ``None``
        align_corners (bool): Whether align corners when resizing heatmaps.
            Defaults to ``False``
        mode (str): Aggregation mode in one of the following:

            - ``'average'``: Get average of heatmaps. All heatmaps mush have
                the same channel number
            - ``'concat'``: Concate the heatmaps at the channel dim
    """

    if mode not in {"average", "concat"}:
        raise ValueError(f"Invalid aggregation mode `{mode}`")

    if size is None:
        h, w = heatmaps[0].shape[2:4]
    else:
        w, h = size

    for i, _heatmaps in enumerate(heatmaps):
        assert _heatmaps.ndim == 4
        if mode == "average":
            assert _heatmaps.shape[:2] == heatmaps[0].shape[:2]
        else:
            assert _heatmaps.shape[0] == heatmaps[0].shape[0]

        if _heatmaps.shape[2:4] != (h, w):
            heatmaps[i] = F.interpolate(_heatmaps, size=(h, w), mode="bilinear", align_corners=align_corners)

    if mode == "average":
        output = sum(heatmaps).div(len(heatmaps))
    elif mode == "concat":
        output = torch.cat(heatmaps, dim=1)
    else:
        raise ValueError()

    return output


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-processing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def _taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] - heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px])
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def keypoints_from_heatmaps(
    heatmaps,
    center,
    scale,
    unbiased=False,
    post_process="default",
    kernel=11,
    valid_radius_factor=0.0546875,
    use_udp=False,
    target_type="GaussianHeatmap",
):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, "megvii"]
    if post_process in ["megvii", "unbiased"]:
        assert kernel > 0
    if use_udp:
        assert not post_process == "megvii"

    # normalize configs
    if post_process is False:
        warnings.warn("post_process=False is deprecated, " "please use post_process=None instead", DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                "post_process=True, unbiased=True is deprecated," " please use post_process='unbiased' instead",
                DeprecationWarning,
            )
            post_process = "unbiased"
        else:
            warnings.warn(
                "post_process=True, unbiased=False is deprecated, " "please use post_process='default' instead",
                DeprecationWarning,
            )
            post_process = "default"
    elif post_process == "default":
        if unbiased is True:
            warnings.warn(
                "unbiased=True is deprecated, please use " "post_process='unbiased' instead", DeprecationWarning
            )
            post_process = "unbiased"

    # start processing
    if post_process == "megvii":
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == "GaussianHeatMap".lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == "CombinedTarget".lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError("target_type should be either " "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == "unbiased":  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array(
                            [heatmap[py][px + 1] - heatmap[py][px - 1], heatmap[py + 1][px] - heatmap[py - 1][px]]
                        )
                        preds[n][k] += np.sign(diff) * 0.25
                        if post_process == "megvii":
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == "megvii":
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals
