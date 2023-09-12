# model settings

model = dict(
    type='SipMaskMem',
    pretrained='~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    # bbox_head=dict(
    #     type='SipMaskHead',
    #     num_classes=41,
    #     in_channels=256,
    #     stacked_convs=3,
    #     feat_channels=256,
    #     strides=[8, 16, 32, 64],
    #     regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
    # pretrained='/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/ViT-b_sipmask/mae_pretrain_vit_b.pth',
    # backbone=dict(
    #     type='ViT',
    #     patch_size=16, 
    #     in_chans=3,
    #     embed_dim=768, 
    #     depth=12, 
    #     num_heads=12,
    #     mlp_ratio=4., 
    #     use_checkpoint=True
    # ),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[768, 768, 768],
    #     out_channels=256,
    #     start_level=0,
    #     add_extra_convs=True,
    #     extra_convs_on_inputs=False,  
    #     num_outs=4,
    #     relu_before_extra_convs=True
    # ),
    bbox_head=dict(
        type='CNPContrastHead',
        num_classes=26,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64],
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_box=dict(type='GIoULoss', 
                     loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True, 
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='BCELoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        #memory and CNP settings
        memory_cfg=dict(
            kdim=128, 
            moving_average_rate=0.999,
            update_mode='top1'),
        center_sampling=True,
        center_sample_radius=1.5))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='uncert'
        ),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    min_bbox_size=0,
    score_thr=0.02,
    nms=dict(type='nms', iou_thr=0.65),
    max_pre_nms=200,
    max_per_img=10)


# dataset settings
dataset_type = 'OVISDataset'
data_root = '/media/data/coky/OVIS/data/ovis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_train.json',
        img_prefix=data_root + 'train',
        img_scale=[(640, 360), (960, 480)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_valid.json',
        img_prefix=data_root + 'valid',
        img_scale=(960, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_valid.json',
        img_prefix=data_root + 'valid',
        img_scale=(960, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
    )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 80,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
cnp_mem_config = None

# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/sipmask_memcnp_ovis_local_bs8'
load_from = None
resume_from = None
workflow = [('train', 1)]
seed = 620