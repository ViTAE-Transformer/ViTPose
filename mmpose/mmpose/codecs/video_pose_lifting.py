# Copyright (c) OpenMMLab. All rights reserved.

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np

from mmpose.registry import KEYPOINT_CODECS
from .base import BaseKeypointCodec


@KEYPOINT_CODECS.register_module()
class VideoPoseLifting(BaseKeypointCodec):
    r"""Generate keypoint coordinates for pose lifter.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - pose-lifitng target dimension: C

    Args:
        num_keypoints (int): The number of keypoints in the dataset.
        zero_center: Whether to zero-center the target around root. Default:
            ``True``.
        root_index (Union[int, List]): Root keypoint index in the pose.
            Default: 0.
        remove_root (bool): If true, remove the root keypoint from the pose.
            Default: ``False``.
        save_index (bool): If true, store the root position separated from the
            original pose, only takes effect if ``remove_root`` is ``True``.
            Default: ``False``.
        reshape_keypoints (bool): If true, reshape the keypoints into shape
            (-1, N). Default: ``True``.
        concat_vis (bool): If true, concat the visibility item of keypoints.
            Default: ``False``.
        normalize_camera (bool): Whether to normalize camera intrinsics.
            Default: ``False``.
    """

    auxiliary_encode_keys = {
        'lifting_target', 'lifting_target_visible', 'camera_param'
    }

    instance_mapping_table = dict(
        lifting_target='lifting_target',
        lifting_target_visible='lifting_target_visible',
    )
    label_mapping_table = dict(
        trajectory_weights='trajectory_weights',
        lifting_target_label='lifting_target_label',
        lifting_target_weight='lifting_target_weight')

    def __init__(self,
                 num_keypoints: int,
                 zero_center: bool = True,
                 root_index: Union[int, List] = 0,
                 remove_root: bool = False,
                 save_index: bool = False,
                 reshape_keypoints: bool = True,
                 concat_vis: bool = False,
                 normalize_camera: bool = False):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.zero_center = zero_center
        if isinstance(root_index, int):
            root_index = [root_index]
        self.root_index = root_index
        self.remove_root = remove_root
        self.save_index = save_index
        self.reshape_keypoints = reshape_keypoints
        self.concat_vis = concat_vis
        self.normalize_camera = normalize_camera

    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: Optional[np.ndarray] = None,
               lifting_target: Optional[np.ndarray] = None,
               lifting_target_visible: Optional[np.ndarray] = None,
               camera_param: Optional[dict] = None) -> dict:
        """Encoding keypoints from input image space to normalized space.

        Args:
            keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D).
            keypoints_visible (np.ndarray, optional): Keypoint visibilities in
                shape (N, K).
            lifting_target (np.ndarray, optional): 3d target coordinate in
                shape (T, K, C).
            lifting_target_visible (np.ndarray, optional): Target coordinate in
                shape (T, K, ).
            camera_param (dict, optional): The camera parameter dictionary.

        Returns:
            encoded (dict): Contains the following items:

                - keypoint_labels (np.ndarray): The processed keypoints in
                  shape like (N, K, D) or (K * D, N).
                - keypoint_labels_visible (np.ndarray): The processed
                  keypoints' weights in shape (N, K, ) or (N-1, K, ).
                - lifting_target_label: The processed target coordinate in
                  shape (K, C) or (K-1, C).
                - lifting_target_weight (np.ndarray): The target weights in
                  shape (K, ) or (K-1, ).
                - trajectory_weights (np.ndarray): The trajectory weights in
                  shape (K, ).

                In addition, there are some optional items it may contain:

                - target_root (np.ndarray): The root coordinate of target in
                  shape (C, ). Exists if ``zero_center`` is ``True``.
                - target_root_removed (bool): Indicate whether the root of
                  pose-lifitng target is removed. Exists if
                  ``remove_root`` is ``True``.
                - target_root_index (int): An integer indicating the index of
                  root. Exists if ``remove_root`` and ``save_index``
                  are ``True``.
                - camera_param (dict): The updated camera parameter dictionary.
                  Exists if ``normalize_camera`` is ``True``.
        """
        if keypoints_visible is None:
            keypoints_visible = np.ones(keypoints.shape[:2], dtype=np.float32)

        if lifting_target is None:
            lifting_target = [keypoints[0]]

        # set initial value for `lifting_target_weight`
        # and `trajectory_weights`
        if lifting_target_visible is None:
            lifting_target_visible = np.ones(
                lifting_target.shape[:-1], dtype=np.float32)
            lifting_target_weight = lifting_target_visible
            trajectory_weights = (1 / lifting_target[:, 2])
        else:
            valid = lifting_target_visible > 0.5
            lifting_target_weight = np.where(valid, 1., 0.).astype(np.float32)
            trajectory_weights = lifting_target_weight

        if camera_param is None:
            camera_param = dict()

        encoded = dict()

        lifting_target_label = lifting_target.copy()
        # Zero-center the target pose around a given root keypoint
        if self.zero_center:
            assert (lifting_target.ndim >= 2 and
                    lifting_target.shape[-2] > max(self.root_index)), \
                f'Got invalid joint shape {lifting_target.shape}'

            root = np.mean(lifting_target[..., self.root_index, :], axis=-2)
            lifting_target_label -= root[..., np.newaxis, :]
            encoded['target_root'] = root

            if self.remove_root and len(self.root_index) == 1:
                root_index = self.root_index[0]
                lifting_target_label = np.delete(
                    lifting_target_label, root_index, axis=-2)
                lifting_target_visible = np.delete(
                    lifting_target_visible, root_index, axis=-2)
                assert lifting_target_weight.ndim in {
                    2, 3
                }, (f'Got invalid lifting target weights shape '
                    f'{lifting_target_weight.shape}')

                axis_to_remove = -2 if lifting_target_weight.ndim == 3 else -1
                lifting_target_weight = np.delete(
                    lifting_target_weight, root_index, axis=axis_to_remove)
                # Add a flag to avoid latter transforms that rely on the root
                # joint or the original joint index
                encoded['target_root_removed'] = True

                # Save the root index for restoring the global pose
                if self.save_index:
                    encoded['target_root_index'] = root_index

        # Normalize the 2D keypoint coordinate with image width and height
        _camera_param = deepcopy(camera_param)
        assert 'w' in _camera_param and 'h' in _camera_param, (
            'Camera parameter `w` and `h` should be provided.')

        center = np.array([0.5 * _camera_param['w'], 0.5 * _camera_param['h']],
                          dtype=np.float32)
        scale = np.array(0.5 * _camera_param['w'], dtype=np.float32)

        keypoint_labels = (keypoints - center) / scale

        assert keypoint_labels.ndim in {
            2, 3
        }, (f'Got invalid keypoint labels shape {keypoint_labels.shape}')
        if keypoint_labels.ndim == 2:
            keypoint_labels = keypoint_labels[None, ...]

        if self.normalize_camera:
            assert 'f' in _camera_param and 'c' in _camera_param, (
                'Camera parameter `f` and `c` should be provided.')
            _camera_param['f'] = _camera_param['f'] / scale
            _camera_param['c'] = (_camera_param['c'] - center[:, None]) / scale
            encoded['camera_param'] = _camera_param

        if self.concat_vis:
            keypoints_visible_ = keypoints_visible
            if keypoints_visible.ndim == 2:
                keypoints_visible_ = keypoints_visible[..., None]
            keypoint_labels = np.concatenate(
                (keypoint_labels, keypoints_visible_), axis=2)

        if self.reshape_keypoints:
            N = keypoint_labels.shape[0]
            keypoint_labels = keypoint_labels.transpose(1, 2, 0).reshape(-1, N)

        encoded['keypoint_labels'] = keypoint_labels
        encoded['keypoints_visible'] = keypoints_visible
        encoded['lifting_target_label'] = lifting_target_label
        encoded['lifting_target_weight'] = lifting_target_weight
        encoded['trajectory_weights'] = trajectory_weights

        return encoded

    def decode(self,
               encoded: np.ndarray,
               target_root: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from normalized space to input image
        space.

        Args:
            encoded (np.ndarray): Coordinates in shape (N, K, C).
            target_root (np.ndarray, optional): The pose-lifitng target root
                coordinate. Default: ``None``.

        Returns:
            keypoints (np.ndarray): Decoded coordinates in shape (N, K, C).
            scores (np.ndarray): The keypoint scores in shape (N, K).
        """
        keypoints = encoded.copy()

        if target_root is not None and target_root.size > 0:
            keypoints = keypoints + target_root
            if self.remove_root and len(self.root_index) == 1:
                keypoints = np.insert(
                    keypoints, self.root_index, target_root, axis=1)
        scores = np.ones(keypoints.shape[:-1], dtype=np.float32)

        return keypoints, scores
