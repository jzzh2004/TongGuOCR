# model settings
checkpoint = './weights/layout/hornet_tiny_gf.pth'  # noqa
angle_version = 'le90'
model = dict(
    type='HDCDetector_Infer',
    angle_version=angle_version,
    backbone=dict(out_indices=(0, 1, 2, 3),
                  num_stages=4,
                  drop_path_rate=0.2,
                  init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    neck=dict(in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            scales=[8],
            ratios=[
                0.03125, 0.0625, 0.0125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0,
                32.0
            ],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(angle_range=angle_version,
                        target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        num_stages=3,
        angle_version=angle_version,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(roi_layer=dict(type='RoIAlignRotated',
                                               out_size=7,
                                               sample_num=2,
                                               clockwise=True),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(in_channels=256,
                       fc_out_channels=1024,
                       roi_feat_size=7,
                       num_classes=1,
                       bbox_coder=dict(angle_range=angle_version,
                                       norm_factor=None,
                                       edge_swap=True,
                                       proj_xy=True,
                                       target_means=(.0, .0, .0, .0, .0),
                                       target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                       reg_class_agnostic=True,
                       loss_cls=dict(use_sigmoid=False, loss_weight=1.0),
                       loss_bbox=dict(beta=1.0, loss_weight=1.0)),
        mask_roi_extractor=dict(roi_layer=dict(type='RoIAlignRotated',
                                               out_size=14,
                                               clockwise=True),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(num_convs=4,
                       in_channels=256,
                       conv_out_channels=256,
                       num_classes=1,
                       loss_mask=dict(use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.7,
                               neg_iou_thr=0.3,
                               min_pos_iou=0.3,
                               match_low_quality=True,
                               ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler',
                              num=256,
                              pos_fraction=0.5,
                              neg_pos_ub=-1,
                              add_gt_as_proposals=False),
                 allowed_border=0,
                 pos_weight=-1,
                 debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(type='MaxIoUAssigner',
                              pos_iou_thr=0.5,
                              neg_iou_thr=0.5,
                              min_pos_iou=0.5,
                              match_low_quality=False,
                              iou_calculator=dict(type='RBboxOverlaps2D'),
                              ignore_iof_thr=-1),
                sampler=dict(
                    # type='OHEMSampler',
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(type='MaxIoUAssigner',
                              pos_iou_thr=0.6,
                              neg_iou_thr=0.6,
                              min_pos_iou=0.6,
                              match_low_quality=False,
                              iou_calculator=dict(type='RBboxOverlaps2D'),
                              ignore_iof_thr=-1),
                sampler=dict(
                    # type='OHEMSampler',
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(type='MaxIoUAssigner',
                              pos_iou_thr=0.7,
                              neg_iou_thr=0.7,
                              min_pos_iou=0.7,
                              match_low_quality=False,
                              iou_calculator=dict(type='RBboxOverlaps2D'),
                              ignore_iof_thr=-1),
                sampler=dict(
                    # type='OHEMSampler',
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(rpn=dict(nms_pre=1000,
                           max_per_img=1000,
                           nms=dict(type='nms', iou_threshold=0.7),
                           min_bbox_size=0),
                  rcnn=dict(nms_pre=2000,
                            min_bbox_size=0,
                            score_thr=0.05,
                            nms=dict(iou_thr=0.5),
                            max_per_img=450,
                            mask_thr_binary=0.5)))

# dataset settings
dataset_type = 'OCRDataset'
data_root = './data/hdc/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RResize',
        img_scale=[(2000, 800), (2000, 840), (2000, 880), (2000, 920),
                   (2000, 960), (2000, 1000), (2000, 1040), (2000, 1080),
                   (2000, 1120), (2000, 1160), (2000, 1200), (2000, 1240),
                   (2000, 1280), (2000, 1320), (2000, 1360),
                   (2000, 1400)],  #(w,h)
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RRandomFlip', flip_ratio=0.5),
    # dict(type='PolyRandomRotate',
    #      rotate_ratio=1.0,
    #      mode='range',
    #      angles_range=30,
    #      auto_bound=False,
    #      rect_classes=None,
    #      version=angle_version),
    dict(
        type='RRandomCrop',
        crop_size=(0.8, 0.5),
        crop_type='relative',
        allow_negative_crop=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(1600, 1600),
         flip=False,
         transforms=[
             dict(type='RResize'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img'])
         ])
]
data = dict(samples_per_gpu=1,
            workers_per_gpu=4,
            train=dict(type=dataset_type,
                       ann_file=data_root + 'train/train_rotate.json',
                       img_prefix=data_root + 'train/img/',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     ann_file=data_root + 'val/val_rotate.json',
                     img_prefix=data_root + 'val/img/',
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      ann_file=data_root + 'val/val_rotate.json',
                      img_prefix=data_root + 'val/img/',
                      pipeline=test_pipeline))

# evaluation
evaluation = dict(greater_keys=[
    'hmean-iou:recall', 'hmean-iou:precision', 'hmean-iou:hmean'
],
                  save_best='hmean-iou:hmean',
                  interval=1,
                  metric='hmean-iou')
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
# paramwise_cfg=dict(
#     custom_keys={
#         'absolute_pos_embed': dict(decay_mult=0.),
#         'relative_position_bias_table': dict(decay_mult=0.),
#         'norm': dict(decay_mult=0.)
#     }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=0.001,
                 step=[80, 128])
runner = dict(type='EpochBasedRunner', max_epochs=160)

checkpoint_config = dict(interval=40, max_keep_ckpts=2)  #!!!

# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
