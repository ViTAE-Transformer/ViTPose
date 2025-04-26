# Copyright (c) OpenMMLab. All rights reserved.

from typing import Optional, Tuple

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec
from .msra_heatmap import MSRAHeatmap
from .regression_label import RegressionLabel


@KEYPOINT_CODECS.register_module()
class IntegralRegressionLabel(BaseKeypointCodec):
    """Generate keypoint coordinates and normalized heatmaps. See the paper:
    `DSNT`_ by Nibali et al(2018).

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [w, h]

    Encoded:

        - keypoint_labels (np.ndarray): The normalized regression labels in
            shape (N, K, D) where D is 2 for 2d coordinates
        - heatmaps (np.ndarray): The generated heatmap in shape (K, H, W) where
            [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        input_size (tuple): Input image size in [w, h]
        heatmap_size (tuple): Heatmap size in [W, H]
        sigma (float): The sigma value of the Gaussian heatmap
        unbiased (bool): Whether use unbiased method (DarkPose) in ``'msra'``
            encoding. See `Dark Pose`_ for details. Defaults to ``False``
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation in DarkPose. The kernel size and sigma should follow
            the expirical formula :math:`sigma = 0.3*((ks-1)*0.5-1)+0.8`.
            Defaults to 11
        normalize (bool): Whether to normalize the heatmaps. Defaults to True.

    .. _`DSNT`: https://arxiv.org/abs/1801.07372
    """

    label_mapping_table = dict(
        keypoint_labels='keypoint_labels',
        keypoint_weights='keypoint_weights',
    )
    field_mapping_table = dict(heatmaps='heatmaps', )

    def __init__(self,
                 input_size: Tuple[int, int],
                 heatmap_size: Tuple[int, int],
                 sigma: float,
                 unbiased: bool = False,
                 blur_kernel_size: int = 11,
                 normalize: bool = True) -> None:
        super().__init__()

        self.heatmap_codec = MSRAHeatmap(input_size, heatmap_size, sigma,
                                         unbiased, blur_kernel_size)
        self.keypoint_codec = RegressionLabel(input_size)
        self.normalize = normalize

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None) -> dict:
        """Encoding keypoints to regression labels and heatmaps.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
            keypoints_visible (np.ndarray): Keypoint visibilities in shape
                (N, K)

        Returns:
            dict:
            - keypoint_labels (np.ndarray): The normalized regression labels in
                shape (N, K, D) where D is 2 for 2d coordinates
            - heatmaps (np.ndarray): The generated heatmap in shape
                (K, H, W) where [W, H] is the `heatmap_size`
            - keypoint_weights (np.ndarray): The target weights in shape
                (N, K)
        """
        encoded_hm = self.heatmap_codec.encode(keypoints, keypoints_visible)
        encoded_kp = self.keypoint_codec.encode(keypoints, keypoints_visible)

        heatmaps = encoded_hm['heatmaps']
        keypoint_labels = encoded_kp['keypoint_labels']
        keypoint_weights = encoded_kp['keypoint_weights']

        if self.normalize:
            val_sum = heatmaps.sum(axis=(-1, -2)).reshape(-1, 1, 1) + 1e-24
            heatmaps = heatmaps / val_sum

        encoded = dict(
            keypoint_labels=keypoint_labels,
            heatmaps=heatmaps,
            keypoint_weights=keypoint_weights)

        return encoded

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, D)

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """

        keypoints, scores = self.keypoint_codec.decode(encoded)

        return keypoints, scores
