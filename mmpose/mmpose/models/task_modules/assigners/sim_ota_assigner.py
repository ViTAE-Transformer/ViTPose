# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.registry import TASK_UTILS
from mmpose.utils.typing import ConfigType

INF = 100000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class SimOTAAssigner:
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (float): Radius of center area to determine
            if a prior is in the center of a gt. Defaults to 2.5.
        candidate_topk (int): Top-k ious candidates to calculate dynamic-k.
            Defaults to 10.
        iou_weight (float): Weight of bbox iou cost. Defaults to 3.0.
        cls_weight (float): Weight of classification cost. Defaults to 1.0.
        oks_weight (float): Weight of keypoint OKS cost. Defaults to 3.0.
        vis_weight (float): Weight of keypoint visibility cost. Defaults to 0.0
        dynamic_k_indicator (str): Cost type for calculating dynamic-k,
            either 'iou' or 'oks'. Defaults to 'iou'.
        use_keypoints_for_center (bool): Whether to use keypoints to determine
            if a prior is in the center of a gt. Defaults to False.
        iou_calculator (dict): Config of IoU calculation method.
            Defaults to dict(type='BBoxOverlaps2D').
        oks_calculator (dict): Config of OKS calculation method.
            Defaults to dict(type='PoseOKS').
    """

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 oks_weight: float = 3.0,
                 vis_weight: float = 0.0,
                 dynamic_k_indicator: str = 'iou',
                 use_keypoints_for_center: bool = False,
                 iou_calculator: ConfigType = dict(type='BBoxOverlaps2D'),
                 oks_calculator: ConfigType = dict(type='PoseOKS')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.oks_weight = oks_weight
        self.vis_weight = vis_weight
        assert dynamic_k_indicator in ('iou', 'oks'), f'the argument ' \
            f'`dynamic_k_indicator` should be either \'iou\' or \'oks\', ' \
            f'but got {dynamic_k_indicator}'
        self.dynamic_k_indicator = dynamic_k_indicator

        self.use_keypoints_for_center = use_keypoints_for_center
        self.iou_calculator = TASK_UTILS.build(iou_calculator)
        self.oks_calculator = TASK_UTILS.build(oks_calculator)

    def assign(self, pred_instances: InstanceData, gt_instances: InstanceData,
               **kwargs) -> dict:
        """Assign gt to priors using SimOTA.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
        Returns:
            dict: Assignment result containing assigned gt indices,
                max iou overlaps, assigned labels, etc.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_keypoints = gt_instances.keypoints
        gt_keypoints_visible = gt_instances.keypoints_visible
        gt_areas = gt_instances.areas
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        keypoints = pred_instances.keypoints
        keypoints_visible = pred_instances.keypoints_visible
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return dict(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes, gt_keypoints, gt_keypoints_visible)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_pred_kpts = keypoints[valid_mask]
        valid_pred_kpts_vis = keypoints_visible[valid_mask]

        num_valid = valid_decoded_bbox.size(0)
        if num_valid == 0:
            # No valid bboxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                      -1,
                                                      dtype=torch.long)
            return dict(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels)

        cost_matrix = (~is_in_boxes_and_center) * INF

        # calculate iou
        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        if self.iou_weight > 0:
            iou_cost = -torch.log(pairwise_ious + EPS)
            cost_matrix = cost_matrix + iou_cost * self.iou_weight

        # calculate oks
        if self.oks_weight > 0 or self.dynamic_k_indicator == 'oks':
            pairwise_oks = self.oks_calculator(
                valid_pred_kpts.unsqueeze(1),  # [num_valid, 1, k, 2]
                target=gt_keypoints.unsqueeze(0),  # [1, num_gt, k, 2]
                target_weights=gt_keypoints_visible.unsqueeze(
                    0),  # [1, num_gt, k]
                areas=gt_areas.unsqueeze(0),  # [1, num_gt]
            )  # -> [num_valid, num_gt]

            oks_cost = -torch.log(pairwise_oks + EPS)
            cost_matrix = cost_matrix + oks_cost * self.oks_weight

        # calculate cls
        if self.cls_weight > 0:
            gt_onehot_label = (
                F.one_hot(gt_labels.to(torch.int64),
                          pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                              num_valid, 1, 1))
            valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(
                1, num_gt, 1)
            # disable AMP autocast to avoid overflow
            with torch.cuda.amp.autocast(enabled=False):
                cls_cost = (
                    F.binary_cross_entropy(
                        valid_pred_scores.to(dtype=torch.float32),
                        gt_onehot_label,
                        reduction='none',
                    ).sum(-1).to(dtype=valid_pred_scores.dtype))
            cost_matrix = cost_matrix + cls_cost * self.cls_weight
        # calculate vis
        if self.vis_weight > 0:
            valid_pred_kpts_vis = valid_pred_kpts_vis.unsqueeze(1).repeat(
                1, num_gt, 1)  # [num_valid, 1, k]
            gt_kpt_vis = gt_keypoints_visible.unsqueeze(
                0).float()  # [1, num_gt, k]
            with torch.cuda.amp.autocast(enabled=False):
                vis_cost = (
                    F.binary_cross_entropy(
                        valid_pred_kpts_vis.to(dtype=torch.float32),
                        gt_kpt_vis.repeat(num_valid, 1, 1),
                        reduction='none',
                    ).sum(-1).to(dtype=valid_pred_kpts_vis.dtype))
            cost_matrix = cost_matrix + vis_cost * self.vis_weight

        if self.dynamic_k_indicator == 'iou':
            matched_pred_ious, matched_gt_inds = \
                self.dynamic_k_matching(
                    cost_matrix, pairwise_ious, num_gt, valid_mask)
        elif self.dynamic_k_indicator == 'oks':
            matched_pred_ious, matched_gt_inds = \
                self.dynamic_k_matching(
                    cost_matrix, pairwise_oks, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious.to(max_overlaps)
        return dict(
            num_gts=num_gt,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)

    def get_in_gt_and_in_center_info(
        self,
        priors: Tensor,
        gt_bboxes: Tensor,
        gt_keypoints: Optional[Tensor] = None,
        gt_keypoints_visible: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Get the information of which prior is in gt bboxes and gt center
        priors."""
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        if self.use_keypoints_for_center and gt_keypoints_visible is not None:
            gt_kpts_cts = (gt_keypoints * gt_keypoints_visible.unsqueeze(-1)
                           ).sum(dim=-2) / gt_keypoints_visible.sum(
                               dim=-1, keepdims=True).clip(min=0)
            gt_kpts_cts = gt_kpts_cts.to(gt_bboxes)
            valid_mask = gt_keypoints_visible.sum(-1) > 0
            gt_cxs[valid_mask] = gt_kpts_cts[valid_mask][..., 0]
            gt_cys[valid_mask] = gt_kpts_cts[valid_mask][..., 1]

        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = (
            is_in_gts[is_in_gts_or_centers, :]
            & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost: Tensor, pairwise_ious: Tensor,
                           num_gt: int,
                           valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets."""
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
