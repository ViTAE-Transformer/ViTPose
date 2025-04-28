_base_ = ["../../../../_base_/default_runtime.py", "../../../../_base_/datasets/coco_wholebody.py"]
evaluation = dict(interval=10, metric="mAP", save_best="AP")

optimizer = dict(
    type="Adam",
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[170, 200])
total_epochs = 210
channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)),
)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    # pretrained=None,
    backbone=dict(
        type="ViT",
        img_size=(256, 192),
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    head=dict(
        type="TopdownHeatmapSimpleHead",
        in_channels=1280,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            final_conv_kernel=1,
        ),
        out_channels=channel_cfg["num_output_channels"],
        loss_keypoint=dict(type="JointsMSELoss", use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(flip_test=True, post_process="default", shift_heatmap=True, modulate_kernel=11),
)

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg["num_output_channels"],
    num_joints=channel_cfg["dataset_joints"],
    dataset_channel=channel_cfg["dataset_channel"],
    inference_channel=channel_cfg["inference_channel"],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file="data/coco/person_detection_results/" "COCO_val2017_detections_AP_H_56_person.json",
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TopDownRandomFlip", flip_prob=0.5),
    dict(type="TopDownHalfBodyTransform", num_joints_half_body=8, prob_half_body=0.3),
    dict(type="TopDownGetRandomScaleRotation", rot_factor=40, scale_factor=0.5),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type="TopDownGenerateTarget", sigma=2),
    dict(
        type="Collect",
        keys=["img", "target", "target_weight"],
        meta_keys=[
            "image_file",
            "joints_3d",
            "joints_3d_visible",
            "center",
            "scale",
            "rotation",
            "bbox_score",
            "flip_pairs",
        ],
    ),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TopDownAffine"),
    dict(type="ToTensor"),
    dict(type="NormalizeTensor", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=["image_file", "center", "scale", "rotation", "bbox_score", "flip_pairs"],
    ),
]

test_pipeline = val_pipeline

data_root = "data/coco"
data_mode = "topdown"

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type="TopDownCocoWholeBodyDataset",
        ann_file=f"{data_root}/annotations/coco_wholebody_train_v1.0.json",
        img_prefix=f"{data_root}/train2017/",
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type="TopDownCocoWholeBodyDataset",
        ann_file=f"{data_root}/annotations/coco_wholebody_val_v1.0.json",
        img_prefix=f"{data_root}/val2017/",
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type="TopDownCocoWholeBodyDataset",
        ann_file=f"{data_root}/annotations/coco_wholebody_val_v1.0.json",
        img_prefix=f"{data_root}/val2017/",
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
)

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_works=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="TopDownCocoWholeBodyDataset",
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/coco_wholebody_train_v1.0.json",
        data_prefix=dict(img="train2017/"),
        data_cfg=data_cfg,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_works=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="TopDownCocoWholeBodyDataset",
        data_root=data_root,
        data_mode=data_mode,
        ann_file="annotations/coco_wholebody_val_v1.0.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        data_cfg=data_cfg,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader
