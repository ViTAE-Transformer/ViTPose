# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from munkres import Munkres
from torch import Tensor

from mmpose.registry import KEYPOINT_CODECS
from mmpose.utils.tensor_utils import to_numpy
from .base import BaseKeypointCodec
from .utils import (batch_heatmap_nms, generate_gaussian_heatmaps,
                    generate_udp_gaussian_heatmaps, refine_keypoints,
                    refine_keypoints_dark_udp)


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _group_keypoints_by_tags(vals: np.ndarray,
                             tags: np.ndarray,
                             locs: np.ndarray,
                             keypoint_order: List[int],
                             val_thr: float,
                             tag_thr: float = 1.0,
                             max_groups: Optional[int] = None) -> np.ndarray:
    """Group the keypoints by tags using Munkres algorithm.

    Note:

        - keypoint number: K
        - candidate number: M
        - tag dimenssion: L
        - coordinate dimension: D
        - group number: G

    Args:
        vals (np.ndarray): The heatmap response values of keypoints in shape
            (K, M)
        tags (np.ndarray): The tags of the keypoint candidates in shape
            (K, M, L)
        locs (np.ndarray): The locations of the keypoint candidates in shape
            (K, M, D)
        keypoint_order (List[int]): The grouping order of the keypoints.
            The groupping usually starts from a keypoints around the head and
            torso, and gruadually moves out to the limbs
        val_thr (float): The threshold of the keypoint response value
        tag_thr (float): The maximum allowed tag distance when matching a
            keypoint to a group. A keypoint with larger tag distance to any
            of the existing groups will initializes a new group
        max_groups (int, optional): The maximum group number. ``None`` means
            no limitation. Defaults to ``None``

    Returns:
        np.ndarray: grouped keypoints in shape (G, K, D+1), where the last
        dimenssion is the concatenated keypoint coordinates and scores.
    """

    tag_k, loc_k, val_k = tags, locs, vals
    K, M, D = locs.shape
    assert vals.shape == tags.shape[:2] == (K, M)
    assert len(keypoint_order) == K

    default_ = np.zeros((K, 3 + tag_k.shape[2]), dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(K):
        idx = keypoint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > val_thr
        tags = tags[mask]  # shape: [M, L]
        joints = joints[mask]  # shape: [M, 3 + L], 3: x, y, val

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            # shape: [M]
            grouped_keys = list(joint_dict.keys())
            # shape: [M, L]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            # shape: [M, M, L]
            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            # shape: [M, M]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)
            diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < tag_thr):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    joint_dict_keys = list(joint_dict.keys())[:max_groups]

    if joint_dict_keys:
        results = np.array([joint_dict[i]
                            for i in joint_dict_keys]).astype(np.float32)
        results = results[..., :D + 1]
    else:
        results = np.empty((0, K, D + 1), dtype=np.float32)
    return results


@KEYPOINT_CODECS.register_module()
class AssociativeEmbedding(BaseKeypointCodec):
    """Encode/decode keypoints with the method introduced in "Associative
    Embedding". This is an asymmetric codec, where the keypoints are
    represented as gaussian heatmaps and position indices during encoding, and
    restored from predicted heatmaps and group tags.

    See the paper `Associative Embedding: End-to-End Learning for Joint
    Detection and Grouping`_ by Newell et al (2017) for details

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - embedding tag dimension: L
        - image size: [w, h]
        - heatmap size: [W, H]

    Encoded:

        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W)
            where [W, H] is the `heatmap_size`
        - keypoint_indices (np.ndarray): The keypoint position indices in shape
            (N, K, 2). Each keypoint's index is [i, v], where i is the position
            index in the heatmap (:math:`i=y*w+x`) and v is the visibility
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        use_udp (bool): Whether use unbiased data processing. See
            `UDP (CVPR 2020)`_ for details. Defaults to ``False``
        decode_keypoint_order (List[int]): The grouping order of the
            keypoint indices. The groupping usually starts from a keypoints
            around the head and torso, and gruadually moves out to the limbs
        decode_keypoint_thr (float): The threshold of keypoint response value
            in heatmaps. Defaults to 0.1
        decode_tag_thr (float): The maximum allowed tag distance when matching
            a keypoint to a group. A keypoint with larger tag distance to any
            of the existing groups will initializes a new group. Defaults to
            1.0
        decode_nms_kernel (int): The kernel size of the NMS during decoding,
            which should be an odd integer. Defaults to 5
        decode_gaussian_kernel (int): The kernel size of the Gaussian blur
            during decoding, which should be an odd integer. It is only used
            when ``self.use_udp==True``. Defaults to 3
        decode_topk (int): The number top-k candidates of each keypoints that
            will be retrieved from the heatmaps during dedocding. Defaults to
            20
        decode_max_instances (int, optional): The maximum number of instances
            to decode. ``None`` means no limitation to the instance number.
            Defaults to ``None``

    .. _`Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping`: https://arxiv.org/abs/1611.05424
    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: Optional[float] = None,
        use_udp: bool = False,
        decode_keypoint_order: List[int] = [],
        decode_nms_kernel: int = 5,
        decode_gaussian_kernel: int = 3,
        decode_keypoint_thr: float = 0.1,
        decode_tag_thr: float = 1.0,
        decode_topk: int = 30,
        decode_center_shift=0.0,
        decode_max_instances: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.use_udp = use_udp
        self.decode_nms_kernel = decode_nms_kernel
        self.decode_gaussian_kernel = decode_gaussian_kernel
        self.decode_keypoint_thr = decode_keypoint_thr
        self.decode_tag_thr = decode_tag_thr
        self.decode_topk = decode_topk
        self.decode_center_shift = decode_center_shift
        self.decode_max_instances = decode_max_instances
        self.decode_keypoint_order = decode_keypoint_order.copy()

        if self.use_udp:
            self.scale_factor = ((np.array(input_size) - 1) /
                                 (np.array(heatmap_size) - 1)).astype(
                                     np.float32)
        else:
            self.scale_factor = (np.array(input_size) /
                                 heatmap_size).astype(np.float32)

        if sigma is None:
            sigma = (heatmap_size[0] * heatmap_size[1])**0.5 / 64
        self.sigma = sigma

    def encode(
        self,
        keypoints: np.ndarray,
        keypoints_visible: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode keypoints into heatmaps and position indices. Note that the
        original keypoint coordinates should be in the input image space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_indices (np.ndarray): The keypoint position indices
                in shape (N, K, 2). Each keypoint's index is [i, v], where i
                is the position index in the heatmap (:math:`i=y*w+x`) and v
                is the visibility
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """

        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        # keypoint coordinates in heatmap
        _keypoints = keypoints / self.scale_factor

        if self.use_udp:
            heatmaps, keypoint_weights = generate_udp_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)
        else:
            heatmaps, keypoint_weights = generate_gaussian_heatmaps(
                heatmap_size=self.heatmap_size,
                keypoints=_keypoints,
                keypoints_visible=keypoints_visible,
                sigma=self.sigma)

        keypoint_indices = self._encode_keypoint_indices(
            heatmap_size=self.heatmap_size,
            keypoints=_keypoints,
            keypoints_visible=keypoints_visible)

        encoded = dict(
            heatmaps=heatmaps,
            keypoint_indices=keypoint_indices,
            keypoint_weights=keypoint_weights)

        return encoded

    def _encode_keypoint_indices(self, heatmap_size: Tuple[int, int],
                                 keypoints: np.ndarray,
                                 keypoints_visible: np.ndarray) -> np.ndarray:
        w, h = heatmap_size
        N, K, _ = keypoints.shape
        keypoint_indices = np.zeros((N, K, 2), dtype=np.int64)

        for n, k in product(range(N), range(K)):
            x, y = (keypoints[n, k] + 0.5).astype(np.int64)
            index = y * w + x
            vis = (keypoints_visible[n, k] > 0.5 and 0 <= x < w and 0 <= y < h)
            keypoint_indices[n, k] = [index, vis]

        return keypoint_indices

    def decode(self, encoded: Any) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def _get_batch_topk(self, batch_heatmaps: Tensor, batch_tags: Tensor,
                        k: int):
        """Get top-k response values from the heatmaps and corresponding tag
        values from the tagging heatmaps.

        Args:
            batch_heatmaps (Tensor): Keypoint detection heatmaps in shape
                (B, K, H, W)
            batch_tags (Tensor): Tagging heatmaps in shape (B, C, H, W), where
                the tag dim C is 2*K when using flip testing, or K otherwise
            k (int): The number of top responses to get

        Returns:
            tuple:
            - topk_vals (Tensor): Top-k response values of each heatmap in
                shape (B, K, Topk)
            - topk_tags (Tensor): The corresponding embedding tags of the
                top-k responses, in shape (B, K, Topk, L)
            - topk_locs (Tensor): The location of the top-k responses in each
                heatmap, in shape (B, K, Topk, 2) where last dimension
                represents x and y coordinates
        """
        B, K, H, W = batch_heatmaps.shape
        L = batch_tags.shape[1] // K

        # shape of topk_val, top_indices: (B, K, TopK)
        topk_vals, topk_indices = batch_heatmaps.flatten(-2, -1).topk(
            k, dim=-1)

        topk_tags_per_kpts = [
            torch.gather(_tag, dim=2, index=topk_indices)
            for _tag in torch.unbind(batch_tags.view(B, L, K, H * W), dim=1)
        ]

        topk_tags = torch.stack(topk_tags_per_kpts, dim=-1)  # (B, K, TopK, L)
        topk_locs = torch.stack([topk_indices % W, topk_indices // W],
                                dim=-1)  # (B, K, TopK, 2)

        return topk_vals, topk_tags, topk_locs

    def _group_keypoints(self, batch_vals: np.ndarray, batch_tags: np.ndarray,
                         batch_locs: np.ndarray):
        """Group keypoints into groups (each represents an instance) by tags.

        Args:
            batch_vals (Tensor): Heatmap response values of keypoint
                candidates in shape (B, K, Topk)
            batch_tags (Tensor): Tags of keypoint candidates in shape
                (B, K, Topk, L)
            batch_locs (Tensor): Locations of keypoint candidates in shape
                (B, K, Topk, 2)

        Returns:
            List[np.ndarray]: Grouping results of a batch, each element is a
            np.ndarray (in shape [N, K, D+1]) that contains the groups
            detected in an image, including both keypoint coordinates and
            scores.
        """

        def _group_func(inputs: Tuple):
            vals, tags, locs = inputs
            return _group_keypoints_by_tags(
                vals,
                tags,
                locs,
                keypoint_order=self.decode_keypoint_order,
                val_thr=self.decode_keypoint_thr,
                tag_thr=self.decode_tag_thr,
                max_groups=self.decode_max_instances)

        _results = map(_group_func, zip(batch_vals, batch_tags, batch_locs))
        results = list(_results)
        return results

    def _fill_missing_keypoints(self, keypoints: np.ndarray,
                                keypoint_scores: np.ndarray,
                                heatmaps: np.ndarray, tags: np.ndarray):
        """Fill the missing keypoints in the initial predictions.

        Args:
            keypoints (np.ndarray): Keypoint predictions in shape (N, K, D)
            keypoint_scores (np.ndarray): Keypint score predictions in shape
                (N, K), in which 0 means the corresponding keypoint is
                missing in the initial prediction
            heatmaps (np.ndarry): Heatmaps in shape (K, H, W)
            tags (np.ndarray): Tagging heatmaps in shape (C, H, W) where
                C=L*K

        Returns:
            tuple:
            - keypoints (np.ndarray): Keypoint predictions with missing
                ones filled
            - keypoint_scores (np.ndarray): Keypoint score predictions with
                missing ones filled
        """

        N, K = keypoints.shape[:2]
        H, W = heatmaps.shape[1:]
        L = tags.shape[0] // K
        keypoint_tags = [tags[k::K] for k in range(K)]

        for n in range(N):
            # Calculate the instance tag (mean tag of detected keypoints)
            _tag = []
            for k in range(K):
                if keypoint_scores[n, k] > 0:
                    x, y = keypoints[n, k, :2].astype(np.int64)
                    x = np.clip(x, 0, W - 1)
                    y = np.clip(y, 0, H - 1)
                    _tag.append(keypoint_tags[k][:, y, x])

            tag = np.mean(_tag, axis=0)
            tag = tag.reshape(L, 1, 1)
            # Search maximum response of the missing keypoints
            for k in range(K):
                if keypoint_scores[n, k] > 0:
                    continue
                dist_map = np.linalg.norm(
                    keypoint_tags[k] - tag, ord=2, axis=0)
                cost_map = np.round(dist_map) * 100 - heatmaps[k]  # H, W
                y, x = np.unravel_index(np.argmin(cost_map), shape=(H, W))
                keypoints[n, k] = [x, y]
                keypoint_scores[n, k] = heatmaps[k, y, x]

        return keypoints, keypoint_scores

    def batch_decode(self, batch_heatmaps: Tensor, batch_tags: Tensor
                     ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Decode the keypoint coordinates from a batch of heatmaps and tagging
        heatmaps. The decoded keypoint coordinates are in the input image
        space.

        Args:
            batch_heatmaps (Tensor): Keypoint detection heatmaps in shape
                (B, K, H, W)
            batch_tags (Tensor): Tagging heatmaps in shape (B, C, H, W), where
                :math:`C=L*K`

        Returns:
            tuple:
            - batch_keypoints (List[np.ndarray]): Decoded keypoint coordinates
                of the batch, each is in shape (N, K, D)
            - batch_scores (List[np.ndarray]): Decoded keypoint scores of the
                batch, each is in shape (N, K). It usually represents the
                confidience of the keypoint prediction
        """
        B, _, H, W = batch_heatmaps.shape
        assert batch_tags.shape[0] == B and batch_tags.shape[2:4] == (H, W), (
            f'Mismatched shapes of heatmap ({batch_heatmaps.shape}) and '
            f'tagging map ({batch_tags.shape})')

        # Heatmap NMS
        batch_heatmaps_peak = batch_heatmap_nms(batch_heatmaps,
                                                self.decode_nms_kernel)

        # Get top-k in each heatmap and and convert to numpy
        batch_topk_vals, batch_topk_tags, batch_topk_locs = to_numpy(
            self._get_batch_topk(
                batch_heatmaps_peak, batch_tags, k=self.decode_topk))

        # Group keypoint candidates into groups (instances)
        batch_groups = self._group_keypoints(batch_topk_vals, batch_topk_tags,
                                             batch_topk_locs)

        # Convert to numpy
        batch_heatmaps_np = to_numpy(batch_heatmaps)
        batch_tags_np = to_numpy(batch_tags)

        # Refine the keypoint prediction
        batch_keypoints = []
        batch_keypoint_scores = []
        batch_instance_scores = []
        for i, (groups, heatmaps, tags) in enumerate(
                zip(batch_groups, batch_heatmaps_np, batch_tags_np)):

            keypoints, scores = groups[..., :-1], groups[..., -1]
            instance_scores = scores.mean(axis=-1)

            if keypoints.size > 0:
                # refine keypoint coordinates according to heatmap distribution
                if self.use_udp:
                    keypoints = refine_keypoints_dark_udp(
                        keypoints,
                        heatmaps,
                        blur_kernel_size=self.decode_gaussian_kernel)
                else:
                    keypoints = refine_keypoints(keypoints, heatmaps)
                keypoints += self.decode_center_shift * \
                    (scores > 0).astype(keypoints.dtype)[..., None]

                # identify missing keypoints
                keypoints, scores = self._fill_missing_keypoints(
                    keypoints, scores, heatmaps, tags)

            batch_keypoints.append(keypoints)
            batch_keypoint_scores.append(scores)
            batch_instance_scores.append(instance_scores)

        # restore keypoint scale
        batch_keypoints = [
            kpts * self.scale_factor for kpts in batch_keypoints
        ]

        return batch_keypoints, batch_keypoint_scores, batch_instance_scores
