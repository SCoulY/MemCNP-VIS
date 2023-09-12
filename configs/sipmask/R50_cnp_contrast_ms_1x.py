# model settings

model = dict(
    type='SipMaskMem',
    pretrained='https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CNPContrastHead',
        num_classes=41,
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
    min_bbox_size=1,
    score_thr=0.03,
    nms=dict(type='nms', iou_thr=0.5),
    max_pre_nms=200,
    max_per_img=10)


# dataset settings
dataset_type = 'YVISDataset'
data_root = '/media/data/coky/OVIS/data/yvis/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/JPEGImages',
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
        ann_file=data_root + 'annotations/instances_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        img_scale=(960, 480),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
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
        dict(type='TextLoggerHook'),
        dict(type='TBImgLoggerHook', 
             interval=50,
             plot_img_bbox=True,
             plot_memory=True,
             plot_neg_bbox=True,
             )
    ])
cnp_mem_config = None

# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sipmask_swin-s'
load_from = None
resume_from = None
workflow = [('train', 1)]
seed = 620
