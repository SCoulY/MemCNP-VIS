# import os
# import json


# COCO_TO_YTVIS_2019 = {
#     1:1, 2:21, 3:6, 4:21, 5:28, 7:17, 8:29, 9:34, 17:14, 18:8, 19:18, 21:15, 22:32, 23:20, 24:30, 25:22, 35:33, 36:33, 41:5, 42:27, 43:40
# }
# COCO_TO_YTVIS_2021 = {
#     1:26, 2:23, 3:5, 4:23, 5:1, 7:36, 8:37, 9:4, 16:3, 17:6, 18:9, 19:19, 21:7, 22:12, 23:2, 24:40, 25:18, 34:14, 35:31, 36:31, 41:29, 42:33, 43:34
# }

# COCO_TO_OVIS = {
#     1:1, 2:21, 3:25, 4:22, 5:23, 6:25, 8:25, 9:24, 17:3, 18:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11, 
# }

# _root = '/media/data/coky/OVIS/data'

# convert_list = [
#     # (COCO_TO_YTVIS_2019, 
#     #     os.path.join(_root, "coco2017/annotations/instances_train2017.json"),
#     #     os.path.join(_root, "coco2017/annotations/coco2ytvis2019_train.json"), "COCO to YTVIS 2019:"),
#     # (COCO_TO_YTVIS_2019, 
#     #     os.path.join(_root, "coco2017/annotations/instances_val2017.json"),
#     #     os.path.join(_root, "coco2017/annotations/coco2ytvis2019_val.json"), "COCO val to YTVIS 2019:"),
#     # (COCO_TO_YTVIS_2021, 
#     #     os.path.join(_root, "coco2017/annotations/instances_train2017.json"),
#     #     os.path.join(_root, "coco2017/annotations/coco2ytvis2021_train.json"), "COCO to YTVIS 2021:"),
#     (COCO_TO_YTVIS_2021, 
#         os.path.join(_root, "coco2017/annotations/instances_val2017.json"),
#         os.path.join(_root, "coco2017/annotations/coco2ytvis2021_val.json"), "COCO val to YTVIS 2021:"),
#     # (COCO_TO_OVIS, 
#     #     os.path.join(_root, "coco2017/annotations/instances_train2017.json"),
#     #     os.path.join(_root, "coco2017/annotations/coco2ovis_train.json"), "COCO to OVIS:"),
# ]

# for convert_dict, src_path, out_path, msg in convert_list:
#     src_f = open(src_path, "r")
#     out_f = open(out_path, "w")
#     src_json = json.load(src_f)
#     # print(src_json.keys())   dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

#     out_json = {}
#     for k, v in src_json.items():
#         if k != 'annotations':
#             out_json[k] = v

#     converted_item_num = 0
#     out_json['annotations'] = []
#     for anno in src_json['annotations']:
#         if anno["category_id"] not in convert_dict:
#             continue
#         anno["category_id"] = convert_dict[anno["category_id"]]
#         out_json['annotations'].append(anno)
#         converted_item_num += 1

#     json.dump(out_json, out_f)
#     print(msg, converted_item_num, "items converted.")

from mmdet.datasets.utils import get_dataset
from mmcv.visualization import imshow_bboxes
from mmdet.core.utils.misc import tensor2imgs
import torch
if __name__ == '__main__':
    torch.manual_seed(428)
    coco_root = '/media/data/coky/OVIS/data/coco2017/'
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='COCOVIS',
        ann_file=coco_root + 'annotations/coco2ytvis2021_val.json',
        img_prefix=coco_root + 'val2017/',
        img_scale=[(640, 360), (960, 480)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True)
    )

    train_dataset = get_dataset(data['train'])

    for i, data in enumerate(train_dataset):
        pass
        # img1 = tensor2imgs(data['img'].data.unsqueeze(0), **img_norm_cfg)[0]
        # imshow_bboxes(img1, data['gt_bboxes'].data.numpy(), out_file='test_valid.png', show=False)

        # img2 = tensor2imgs(data['img'].data.unsqueeze(0), **img_norm_cfg)[0]
        # imshow_bboxes(img2, data['gt_bboxes_ignore'].data.numpy(), out_file='test_ignore.png', show=False)

        # img1_ref = tensor2imgs(data['ref_img'].data.unsqueeze(0), **img_norm_cfg)[0]
        # imshow_bboxes(img1_ref, data['ref_bboxes'].data.numpy(), out_file='test1_ref.png', show=False)
        # import pdb; pdb.set_trace()
    print(train_dataset.ignore) 
    import numpy as np 
    np.save('ignore.npy', np.asarray(train_dataset.ignore))