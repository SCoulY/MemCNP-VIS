import time
import torch
from mmdet import datasets
from mmdet.datasets import build_dataloader
from mmcv.runner import obj_from_dict

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
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_valid.json',
        img_prefix=data_root + 'valid',
        img_scale=(640, 360),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))

dataset = obj_from_dict(data['train'], datasets, dict(test_mode=False))
data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=1,
            num_gpus=1,
            dist=False,
            shuffle=False)

def intersection(box1, box2):
    # box1: tensor of (x1, y1, x2, y2)
    # box2: tensor of (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]) 
    y2 = min(box1[3], box2[3]) 
    return (torch.floor(x1).int(), torch.floor(y1).int(), 
            torch.ceil(x2).int(), torch.ceil(y2).int())

if __name__ == '__main__':
    overlap = 0
    total_mis_reg_points = 0
    total_correct_points = 0

    for imd_id, data in enumerate(data_loader):
        t1 = time.time()
        img = data['img'].data[0] # tensor of shape (1, 3, H, W)
        gt_mask = data['gt_masks'].data[0][0] # array of shape (N, H, W)
        gt_bboxes = data['gt_bboxes'].data[0][0] # tensor of shape (N,4)
        gt_pids = data['gt_pids'].data[0][0] # tensor of shape (N,1)
        if gt_pids.size(0) < 2:
            continue
        else:
            for i, pid in enumerate(gt_pids[:-1]):
                for j, pid in enumerate(gt_pids[i+1:]):
                    x11, y11, x12, y12 = gt_bboxes[i]
                    h1 = y12 - y11
                    w1 = x12 - x11
                    area1 = h1 * w1
                    x21, y21, x22, y22 = gt_bboxes[j+i+1]
                    h2 = y22 - y21
                    w2 = x22 - x21
                    area2 = h2 * w2
                    if x11<x21<x12 and y11<y21<y12 or \
                        x11<x22<x12 and y11<y22<y12:
                        overlap += 1
                        x1,y1,x2,y2 = intersection(gt_bboxes[i], gt_bboxes[j+i+1]) #intersected box
                        gt_mask1 = gt_mask[i] # array of shape (H, W)
                        gt_mask1[gt_mask1!=0] = i
                        gt_mask2 = gt_mask[j+i+1] # array of shape (H, W)
                        gt_mask2[gt_mask2!=0] = j+i+1
                        occluder_mask = (gt_mask1 + gt_mask2)[y1:y2, x1:x2] # array of shape (y2-y1, x2-x1)
                        reg_box1 = (occluder_mask==i).sum()
                        reg_box2 = (occluder_mask==j+i+1).sum()
                        if area1>area2:
                            mis_reg = reg_box1
                            correct_reg = reg_box2
                        elif area2>area1:
                            mis_reg = reg_box2
                            correct_reg = reg_box1
                        total_mis_reg_points += mis_reg
                        total_correct_points += correct_reg
                        continue
        t2=time.time()
        print('{}/{} time: {}'.format(imd_id, len(data_loader), t2-t1))
    print('overlap: {}'.format(overlap))
    print('total_mis_reg_points: {}'.format(total_mis_reg_points))
    print('total_correct_points: {}'.format(total_correct_points))