import time
import torch
from mmdet import datasets
from mmdet.datasets import build_dataloader
from mmcv.runner import obj_from_dict

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

def cent_coord_expand(cent_coords):
    '''cent_coords: tensor of shape (N,2)'''
    cent_coords_grid = torch.zeros((cent_coords.size(0), cent_coords.size(1), 3, 3))
    for i in range(3):
        for j in range(3):
            cent_coords_grid[:,:,i,j] = torch.stack([cent_coords[:,0] + i - 1, cent_coords[:,1] + j - 1], 1)
    return cent_coords_grid

if __name__ == '__main__':
    overlap = 0
    total_mis_reg_points = 0
    total_correct_points = 0

    total_cent_pixs = 0
    total_hit_pixs = 0
    total_box_pixs = 0
    t1 = time.time()
    for imd_id, data in enumerate(data_loader):
        img = data['img'].data[0] # tensor of shape (1, 3, H, W)
        gt_mask = data['gt_masks'].data[0][0] # array of shape (N, H, W)
        gt_bboxes = data['gt_bboxes'].data[0][0] # tensor of shape (N,4)
        gt_pids = data['gt_pids'].data[0][0] # tensor of shape (N)
        if not gt_pids.size(0):
            continue
        gt_mask_totoal = gt_mask.sum(0) # array of shape (H, W)
        gt_mask_totoal[gt_mask_totoal!=0] = 1
        cent_coords = torch.stack([(gt_bboxes[:,0] + gt_bboxes[:,2])/2, (gt_bboxes[:,1] + gt_bboxes[:,3])/2], 1) # tensor of shape (N,2)
        cent_coords_grid = cent_coord_expand(cent_coords) # tensor of shape (N,2,3,3)
        cent_coords_grid = cent_coords_grid.permute(0,2,3,1).contiguous().view(-1, 2) # tensor of shape (N*9,2)
        cent_coords_quant = torch.round(cent_coords_grid) # tensor of shape (N*9,2)
        cent_coords_quant[:,0] = torch.clamp(cent_coords_quant[:,0], 0, img.size(3))
        cent_coords_quant[:,1] = torch.clamp(cent_coords_quant[:,1], 0, img.size(2)) 
        hit_pixs = gt_mask_totoal[cent_coords_quant[:,1].long(), cent_coords_quant[:,0].long()].sum()
        cent_pixs = cent_coords_grid.size(0)
        box_pixs = ((gt_bboxes[:,2] - gt_bboxes[:,0]) * (gt_bboxes[:,3] - gt_bboxes[:,1])).sum()
        total_cent_pixs += cent_pixs
        total_hit_pixs += hit_pixs
        total_box_pixs += box_pixs

        # if gt_pids.size(0) < 2:
        #     continue
        # else:
        #     for i, pid in enumerate(gt_pids[:-1]):
        #         for j, pid in enumerate(gt_pids[i+1:]):
        #             x11, y11, x12, y12 = gt_bboxes[i]
        #             h1 = y12 - y11
        #             w1 = x12 - x11
        #             area1 = h1 * w1
        #             x21, y21, x22, y22 = gt_bboxes[j+i+1]
        #             h2 = y22 - y21
        #             w2 = x22 - x21
        #             area2 = h2 * w2
        #             if x11<x21<x12 and y11<y21<y12 or \
        #                 x11<x22<x12 and y11<y22<y12:
        #                 overlap += 1
        #                 x1,y1,x2,y2 = intersection(gt_bboxes[i], gt_bboxes[j+i+1]) #intersected box
        #                 gt_mask1 = gt_mask[i] # array of shape (H, W)
        #                 gt_mask1[gt_mask1!=0] = i
        #                 gt_mask2 = gt_mask[j+i+1] # array of shape (H, W)
        #                 gt_mask2[gt_mask2!=0] = j+i+1
        #                 occluder_mask = (gt_mask1 + gt_mask2)[y1:y2, x1:x2] # array of shape (y2-y1, x2-x1)
        #                 reg_box1 = (occluder_mask==i).sum()
        #                 reg_box2 = (occluder_mask==j+i+1).sum()
        #                 if area1>area2:
        #                     mis_reg = reg_box1
        #                     correct_reg = reg_box2
        #                 elif area2>area1:
        #                     mis_reg = reg_box2
        #                     correct_reg = reg_box1
        #                 total_mis_reg_points += mis_reg
        #                 total_correct_points += correct_reg
        #                 continue

    t2=time.time()
    print('time: {}'.format(t2-t1))
    import pdb; pdb.set_trace()
    sampling_hit_rate = total_hit_pixs / total_cent_pixs
    print('sampling_hit_rate: {}'.format(sampling_hit_rate))
    # print('overlap: {}'.format(overlap))
    # print('total_mis_reg_points: {}'.format(total_mis_reg_points))
    # print('total_correct_points: {}'.format(total_correct_points))
            