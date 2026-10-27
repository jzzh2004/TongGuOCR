# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import ModuleList
from .box_head.hdc_box_head import HDCBBoxHead_v2
from .mask_head.fcn_mask_head import FCNMaskHead
from .roi_extractor import RotatedMultiRoIExtractor
from mmrotate.core import (build_assigner, build_sampler, rbbox2result,
                           rbbox2roi)
from mmrotate.core import obb2poly
import copy


class HDCRoIHead(nn.Module):
    """Cascade roi head including one bbox head and one mask head.
    In inference time,we only use mask head one time
    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages=3,
                 angle_version='le90',
                 stage_loss_weights=[1, 0.5, 0.25],
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__()
        self.angle_version = angle_version
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_bbox_head(bbox_roi_extractor, bbox_head)
        self.init_mask_head(mask_roi_extractor, mask_head)
        self.init_assigner_sampler()

    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(1)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(1)]
        assert len(bbox_roi_extractor) == len(bbox_head) == 1
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(
                RotatedMultiRoIExtractor(**roi_extractor))
            self.bbox_head.append(HDCBBoxHead_v2(**head))

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        self.mask_roi_extractor = ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(1)]
        if not isinstance(mask_roi_extractor, list):
            mask_roi_extractor = [mask_roi_extractor for _ in range(1)]
        assert len(mask_head) == len(mask_roi_extractor) == 1
        for roi_extractor, head in zip(mask_roi_extractor, mask_head):
            self.mask_roi_extractor.append(
                RotatedMultiRoIExtractor(**roi_extractor))
            self.mask_head.append(FCNMaskHead(**head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner = []
        self.bbox_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(
                    build_assigner(rcnn_train_cfg['assigner']))
                self.current_stage = idx
                self.bbox_sampler.append(
                    build_sampler(rcnn_train_cfg['sampler'], context=self))

    def forward(self,
                x,
                img_metas,
                proposal_list,
                rescale=True,
                gt_bboxes=None,
                gt_labels=None,
                gt_bboxes_ignore=None,
                gt_masks=None,
                **kwargs):
        if self.training:
            return self.forward_train(x=x,
                                      img_metas=img_metas,
                                      proposal_list=proposal_list,
                                      gt_bboxes=gt_bboxes,
                                      gt_labels=gt_labels,
                                      gt_bboxes_ignore=gt_bboxes_ignore,
                                      gt_masks=gt_masks,
                                      **kwargs)
        else:
            return self.simple_test(x=x,
                                    proposal_list=proposal_list,
                                    img_metas=img_metas,
                                    rescale=rescale)

    def _bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        stage = 0  # only one box head
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        cls_score, bbox_pred = bbox_head(bbox_feats)

        bbox_results = dict(cls_score=cls_score,
                            bbox_pred=bbox_pred,
                            bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
                            gt_labels, rcnn_train_cfg):
        """Run forward function and calculate loss for box head in training."""
        stage = 0  # only one box head
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_targets = self.bbox_head[stage].get_targets(
            sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
        loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
                                               bbox_results['bbox_pred'], rois,
                                               *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox,
                            rois=rois,
                            bbox_targets=bbox_targets)
        return bbox_results

    def _mask_forward(self, stage, x, rois):
        """Mask head forward function used in both training and testing."""
        stage = 0  # only one mask head
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_pred = mask_head(mask_feats)

        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def poly2bboxes(self, polys):
        x0 = polys[:, ::2].min(dim=1).values.view(-1, 1)
        y0 = polys[:, 1::2].min(dim=1).values.view(-1, 1)
        x1 = polys[:, ::2].max(dim=1).values.view(-1, 1)
        y1 = polys[:, 1::2].max(dim=1).values.view(-1, 1)

        boxes = torch.cat([x0, y0, x1, y1], dim=1)
        return boxes

    def _mask_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_masks,
                            rcnn_train_cfg,
                            bbox_feats=None):
        """Run forward function and calculate loss for mask head in
        training."""
        minrect_sampling_results = []
        for s in sampling_results:
            new_s = copy.deepcopy(s)
            pos_bboxes = obb2poly(s.pos_bboxes, version=self.angle_version)
            neg_bboxes = obb2poly(s.neg_bboxes, version=self.angle_version)
            pos_bboxes = self.poly2bboxes(pos_bboxes)
            neg_bboxes = self.poly2bboxes(neg_bboxes)
            new_s.pos_bboxes = pos_bboxes
            new_s.neg_bboxes = neg_bboxes
            minrect_sampling_results.append(new_s)
        rpos_rois = rbbox2roi([res.pos_bboxes for res in sampling_results])
        stage = 0  # only one mask_head
        mask_results = self._mask_forward(stage, x, rpos_rois)

        mask_targets = self.mask_head[stage].get_targets(
            minrect_sampling_results, gt_masks, rcnn_train_cfg)
        pos_labels = torch.cat(
            [res.pos_gt_labels for res in minrect_sampling_results])
        loss_mask = self.mask_head[stage].loss(mask_results['mask_pred'],
                                               mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[i]
                bbox_sampler = self.bbox_sampler[i]
                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_results = self._bbox_forward_train(i, x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    rcnn_train_cfg)

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (value *
                                          lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                mask_results = self._mask_forward_train(
                    i, x, sampling_results, gt_masks, rcnn_train_cfg,
                    bbox_results['bbox_feats'])
                for name, value in mask_results['loss_mask'].items():
                    losses[f's{i}.{name}'] = (value *
                                              lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                # bbox_targets is a tuple
                roi_labels = bbox_results['bbox_targets'][0]
                with torch.no_grad():
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[0].custom_activation:
                        cls_score = self.bbox_head[0].loss_cls.get_activation(
                            cls_score)

                    # Empty proposal.
                    if cls_score.numel() == 0:
                        break

                    roi_labels = torch.where(
                        roi_labels == self.bbox_head[0].num_classes,
                        cls_score[:, :-1].argmax(1), roi_labels)
                    proposal_list = self.bbox_head[0].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)

        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg
        rois = rbbox2roi(proposal_list).to(x[0].dtype)

        if rois.shape[0] == 0:
            # There is no proposal in the whole batch
            bbox_results = [[
                np.zeros((0, 6), dtype=np.float32)
                for _ in range(self.bbox_head[-1].num_classes)
            ]] * num_imgs

            if self.with_mask:
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
                results = list(zip(bbox_results, segm_results))
            else:
                results = bbox_results

            return results

        for i in range(self.num_stages):

            bbox_results = self._bbox_forward(i, x, rois)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(
                len(proposals) for proposals in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head[0].bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                if self.bbox_head[0].custom_activation:
                    cls_score = [
                        self.bbox_head[0].loss_cls.get_activation(s)
                        for s in cls_score
                    ]
                refine_rois_list = []
                for j in range(num_imgs):
                    if rois[j].shape[0] > 0:
                        bbox_label = cls_score[j][:, :-1].argmax(dim=1)
                        refined_rois = self.bbox_head[0].regress_by_class(
                            rois[j], bbox_label, bbox_pred[j], img_metas[j])
                        refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list)

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_keep_inds = []
        for i in range(num_imgs):
            det_bbox, det_label, keep_ind = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_keep_inds.append(keep_ind)
        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]
        ms_bbox_result['ensemble'] = bbox_results
        # box_end = perf_counter()
        # box_time = (box_end - box_start)
        # print('box time:{}'.format(box_time))

        # mask_start = perf_counter()
        if self.with_mask:
            if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
                mask_classes = self.mask_head[-1].num_classes
                segm_results = [[[] for _ in range(mask_classes)]
                                for _ in range(num_imgs)]
            else:
                if rescale and not isinstance(scale_factors[0], float):
                    rscale_factors = [
                        torch.from_numpy(np.append(scale_factor, 1)).to(
                            det_bboxes[0].device).float()
                        for scale_factor in scale_factors
                    ]
                    scale_factors = [
                        torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                        for scale_factor in scale_factors
                    ]

                polys = [
                    obb2poly(det_bboxes[i][:, :5], version=self.angle_version)
                    for i in range(len(det_bboxes))
                ]
                hbboxes = [
                    self.poly2bboxes(polys[i]) for i in range(len(polys))
                ]

                _bboxes = [
                    hbboxes[i] * scale_factors[i] if rescale else hbboxes[i]
                    for i in range(len(hbboxes))
                ]
                _rbboxes = [
                    det_bboxes[i][:, :5] *
                    rscale_factors[i] if rescale else det_bboxes[i][:, :5]
                    for i in range(len(det_bboxes))
                ]

                mask_rois = rbbox2roi(_rbboxes).to(x[0].dtype)
                num_mask_rois_per_img = tuple(
                    _bbox.size(0) for _bbox in _bboxes)

                mask_results = self._mask_forward(i, x, mask_rois)
                mask_pred = mask_results['mask_pred']
                # split batch mask prediction back to each image
                mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
                aug_masks = [
                    m.sigmoid().cpu().detach().numpy() for m in mask_pred
                ]

                # apply mask post-processing to each image individually
                segm_results = []
                for i in range(num_imgs):
                    if det_bboxes[i].shape[0] == 0:
                        segm_results.append(
                            [[]
                             for _ in range(self.mask_head[-1].num_classes)])
                    else:
                        merged_masks = aug_masks[0]
                        segm_result = self.mask_head[-1].get_seg_masks(
                            merged_masks, _bboxes[i], det_labels[i],
                            rcnn_test_cfg, ori_shapes[i], scale_factors[i],
                            rescale)
                        segm_results.append(segm_result)
            ms_segm_result['ensemble'] = segm_results

        if self.with_mask:
            results = list(
                zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
        else:
            results = ms_bbox_result['ensemble']

        return results
