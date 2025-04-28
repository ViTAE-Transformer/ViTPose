# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData, PixelData
from mmengine.utils import is_list_of

from .bbox.transforms import get_warp_matrix
from .pose_data_sample import PoseDataSample


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    """Merge the given data samples into a single data sample.

    This function can be used to merge the top-down predictions with
    bboxes from the same image. The merged data sample will contain all
    instances from the input data samples, and the identical metainfo with
    the first input data sample.

    Args:
        data_samples (List[:obj:`PoseDataSample`]): The data samples to
            merge

    Returns:
        PoseDataSample: The merged data sample.
    """

    if not is_list_of(data_samples, PoseDataSample):
        raise ValueError('Invalid input type, should be a list of '
                         ':obj:`PoseDataSample`')

    if len(data_samples) == 0:
        warnings.warn('Try to merge an empty list of data samples.')
        return PoseDataSample()

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    if 'gt_instances' in data_samples[0]:
        merged.gt_instances = InstanceData.cat(
            [d.gt_instances for d in data_samples])

    if 'pred_instances' in data_samples[0]:
        merged.pred_instances = InstanceData.cat(
            [d.pred_instances for d in data_samples])

    if 'pred_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].pred_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.pred_fields.heatmaps,
                           data_sample.input_center, data_sample.input_scale,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        pred_fields = PixelData()
        pred_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.pred_fields = pred_fields

    if 'gt_fields' in data_samples[0] and 'heatmaps' in data_samples[
            0].gt_fields:
        reverted_heatmaps = [
            revert_heatmap(data_sample.gt_fields.heatmaps,
                           data_sample.input_center, data_sample.input_scale,
                           data_sample.ori_shape)
            for data_sample in data_samples
        ]

        merged_heatmaps = np.max(reverted_heatmaps, axis=0)
        gt_fields = PixelData()
        gt_fields.set_data(dict(heatmaps=merged_heatmaps))
        merged.gt_fields = gt_fields

    return merged


def revert_heatmap(heatmap, input_center, input_scale, img_shape):
    """Revert predicted heatmap on the original image.

    Args:
        heatmap (np.ndarray or torch.tensor): predicted heatmap.
        input_center (np.ndarray): bounding box center coordinate.
        input_scale (np.ndarray): bounding box scale.
        img_shape (tuple or list): size of original image.
    """
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().detach().numpy()

    ndim = heatmap.ndim
    # [K, H, W] -> [H, W, K]
    if ndim == 3:
        heatmap = heatmap.transpose(1, 2, 0)

    hm_h, hm_w = heatmap.shape[:2]
    img_h, img_w = img_shape
    warp_mat = get_warp_matrix(
        input_center.reshape((2, )),
        input_scale.reshape((2, )),
        rot=0,
        output_size=(hm_w, hm_h),
        inv=True)

    heatmap = cv2.warpAffine(
        heatmap, warp_mat, (img_w, img_h), flags=cv2.INTER_LINEAR)

    # [H, W, K] -> [K, H, W]
    if ndim == 3:
        heatmap = heatmap.transpose(2, 0, 1)

    return heatmap


def split_instances(instances: InstanceData) -> List[InstanceData]:
    """Convert instances into a list where each element is a dict that contains
    information about one instance."""
    results = []

    # return an empty list if there is no instance detected by the model
    if instances is None:
        return results

    for i in range(len(instances.keypoints)):
        result = dict(
            keypoints=instances.keypoints[i].tolist(),
            keypoint_scores=instances.keypoint_scores[i].tolist(),
        )
        if 'bboxes' in instances:
            result['bbox'] = instances.bboxes[i].tolist(),
            if 'bbox_scores' in instances:
                result['bbox_score'] = instances.bbox_scores[i]
        results.append(result)

    return results
