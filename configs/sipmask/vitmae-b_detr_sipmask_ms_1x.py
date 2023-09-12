# model settings
model = dict(
    type='SipMaskDetr',
    pretrained='/media/data/coky/OVIS/CMaskTrack-RCNN/workdir/sipmask_swin-s/swin-s_imagenet_pretrained.pth',
    backbone=dict(
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        frozen_stages=4,
    ),
    neck=dict(
        type='FPDETR',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        encoder_layers=3,
        decoder_layers=3,
        activation='gelu',
        num_queries=100,
        use_checkpoint=True
    ),
    bbox_head=dict(
        type='SipMaskDetrHead',
        num_classes=41,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[4, 8, 16, 32],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            reduction='sum',
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_box=dict(type='GIoULoss', 
                     loss_weight=1.0)
    )
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    max_pre_match=20,
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/JPEGImages',
        img_scale=[(640, 360)],
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
        img_scale=(640, 360),
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
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))

# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-4,
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
             plot_img_bbox=True
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
