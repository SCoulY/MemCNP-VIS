# ------------------------------------------------------------------------
# SeqFormer data loader
# ------------------------------------------------------------------------
# Modified from Deformable VisTR (https://github.com/Epiphqny/VisTR)
# ------------------------------------------------------------------------


"""
 augment coco image to generate a query-reference image pair
"""
import torch
import torch.utils.data

from .custom import CustomDataset
from pycocotools.coco import COCO
from pycocotools.coco import maskUtils
import numpy as np
from PIL import Image
from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from .utils import to_tensor, random_scale
from torchvision import transforms
from mmcv.parallel import DataContainer as DC
import mmcv
import os

class CocoDetection(CustomDataset):
    CLASSES=('airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog', 'duck', 'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox', 'frog', 'giant_panda', 'giraffe', 'horse', 'leopard', 'lizard', 'monkey', 'motorbike', 'mouse', 'parrot', 'person', 'rabbit', 'shark', 'skateboard', 'snake', 'snowboard', 'squirrel', 'surfboard', 'tennis_racket', 'tiger', 'train', 'truck', 'turtle', 'whale', 'zebra')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 resize_keep_ratio=True):

        # prefix of images path
        self.img_prefix = img_prefix
        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg
        # flip ratio
        self.flip_ratio = flip_ratio
        assert flip_ratio >= 0 and flip_ratio <= 1
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        self.resize_keep_ratio = resize_keep_ratio
        
        self.augmenter = TransformWithMask(rotation_range=(-20, 20), perspective_magnitude=0.08,
                                                    hue_magnitude=0.05, brightness_magnitude=0.1,
                                                    contrast_magnitude=0.02, saturation_magnitude=0.02,
                                                    motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                                                    translate_range=(-0.1, 0.1))
                
        self.coco = COCO(ann_file)

        img_ids = []
        for ann in self.coco.anns.values():
            img_ids.append(ann['image_id'])
        self.ids = sorted(list(set(img_ids)))

        # rescale
        self.img_rescale = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        self.bbox_rescale = BboxTransform()
        self.mask_rescale = MaskTransform()
        self.numpy2tensor = Numpy2Tensor()

        self.ignore = [22174, 54607, 97065, 111062, 393221, 435405, 437597] # filter out the image with no valid gt_box

        self.ids = sorted(list(set(self.ids) - set(self.ignore)))
        self._set_group_flag()

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target_all = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_prefix, path)).convert('RGB')

        gt_bboxes_ignore = []
        target = []
        for i, t in enumerate(target_all):
            if t['iscrowd']:
                gt_bboxes_ignore.append(t['bbox'])
            else:
                target.append(t)
        target = {'image_id': img_id, 'annotations': target}

        img, target = self.ConvertCocoPolysToMask(img, target)         
        numpy_masks = target['masks'].numpy()
        valid_ind = numpy_masks.sum(axis=(1, 2))>9
        valid_masks = numpy_masks[valid_ind] #num_gt, h, w


        img_qry = np.asarray(img)
        mask_qry = valid_masks
        box_qry = self.masks_to_boxes(torch.from_numpy(mask_qry)).numpy()


        img_ref, mask_ref = self.augmenter(img, torch.from_numpy(valid_masks))
        # mask_ref = np.stack(mask_ref, axis=0)

        # box_ref = self.masks_to_boxes(torch.from_numpy(mask_ref)).numpy()
        # add co-occurred masks in gt_pids
        gt_pids = []
        gt_labels = []
        box_val = []
        box_ref_val = []
        mask_val = []
        for i in range(mask_qry.shape[0]):
            if mask_qry[i].sum() >= 9 and mask_ref[i].sum() >= 9:
                gt_pids.append(np.count_nonzero(gt_pids)+1)
                gt_labels.append(target['labels'][i])
                box_val.append(box_qry[i])
                box_ref_i = self.masks_to_boxes(mask_ref[i].unsqueeze(0)).numpy()
                box_ref_val.append(box_ref_i)
                mask_val.append(mask_qry[i])
            elif mask_qry[i].sum() >= 9 and mask_ref[i].sum() < 9:
                gt_pids.append(0)
                gt_labels.append(target['labels'][i])
                box_val.append(box_qry[i])
                mask_val.append(mask_qry[i])

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        img_scale = random_scale(self.img_scales)  # sample a scale

        img, img_shape, pad_shape, scale_factor = self.img_rescale(
            img_qry, img_scale, flip, keep_ratio=self.resize_keep_ratio)

        ref_img, ref_img_shape, _, ref_scale_factor = self.img_rescale(
            img_ref, img_scale, flip, keep_ratio=self.resize_keep_ratio)

        gt_bboxes = self.bbox_rescale(np.asarray(box_val), img_shape, scale_factor,
                                        flip)
    
        if box_ref_val:
            ref_bboxes = self.bbox_rescale(np.concatenate(box_ref_val, axis=0), ref_img_shape, ref_scale_factor,
                                            flip)
        else: #if no occurence in ref_img, use gt_bboxes as ref_bboxes but gt_pids = 0
            ref_bboxes = gt_bboxes

        if self.with_mask:
            gt_masks = self.mask_rescale(np.asarray(mask_val), pad_shape,
                                        scale_factor, flip)

        ori_shape = target['size']
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            ref_img=DC(to_tensor(ref_img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)),
            ref_bboxes = DC(to_tensor(ref_bboxes))
        )

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_track:
            data['gt_pids'] = DC(to_tensor(gt_pids))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        return data
    
    def __len__(self):
        return len(self.ids)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_id = self.ids[i]
            w = self.coco.imgs[img_id]['width']
            h = self.coco.imgs[img_id]['height']
            if w / h > 1:
                self.flag[i] = 1

    def convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = maskUtils.frPyObjects(polygons, height, width)
            mask = maskUtils.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks
    
    def ConvertCocoPolysToMask(self, image, target):
        
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)


        classes = [obj['category_id'] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.with_mask:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = self.convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.with_mask:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.with_mask:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

    def masks_to_boxes(self, masks):
        """Compute the bounding boxes around the provided masks
        The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
        Returns a [N, 4] tensors, with the boxes in xyxy format
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)

        h, w = masks.shape[-2:]

        y = torch.arange(0, h, dtype=torch.float, device=masks.device)
        x = torch.arange(0, w, dtype=torch.float, device=masks.device)
        y, x = torch.meshgrid(y, x)

        x_mask = (masks * x.unsqueeze(0))
        x_max = x_mask.flatten(1).max(-1)[0]
        x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        y_mask = (masks * y.unsqueeze(0))
        y_max = y_mask.flatten(1).max(-1)[0]
        y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

        return torch.stack([x_min, y_min, x_max, y_max], 1)



class TransformWithMask(object):
    def __init__(self,rotation_range=(-20, 20), perspective_magnitude=0.08,
                    hue_magnitude=0.01, brightness_magnitude=0.05,
                    contrast_magnitude=0.01, saturation_magnitude=0.01,
                    motion_blur_prob=0.25, motion_blur_kernel_sizes=(9, 11),
                    translate_range=(-0.1, 0.1)):
        self.rotation_range = rotation_range
        self.perspective_magnitude = perspective_magnitude
        self.hue_magnitude = hue_magnitude
        self.brightness_magnitude = brightness_magnitude
        self.contrast_magnitude = contrast_magnitude
        self.saturation_magnitude = saturation_magnitude
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_sizes = motion_blur_kernel_sizes
        self.translate_range = translate_range
        self.color_jitter = transforms.ColorJitter(brightness=self.brightness_magnitude, 
                                    contrast=self.contrast_magnitude, 
                                    saturation=self.saturation_magnitude, 
                                    hue=self.hue_magnitude)

    def perspective(self, img, mask):
        h, w = img.size
        magnitude = self.perspective_magnitude
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2[:, 0] += np.random.uniform(-magnitude, magnitude) * w
        pts2[:, 1] += np.random.uniform(-magnitude, magnitude) * h
        img = transforms.functional.perspective(img, pts1, pts2)
        mask = transforms.functional.perspective(mask, pts1, pts2)
        return img, mask
    
    def motion_blur(self, img):
        img = transforms.functional.gaussian_blur(img, self.motion_blur_kernel_sizes)
        return img
    
    def translate(self, img, mask):
        h, w = img.size
        tx = np.random.uniform(self.translate_range[0], self.translate_range[1]) * w
        ty = np.random.uniform(self.translate_range[0], self.translate_range[1]) * h
        degree = np.random.randint(self.rotation_range[0], self.rotation_range[1])
        img = transforms.functional.affine(img, angle=degree, translate=(tx, ty), scale=1, shear=0)
        mask = transforms.functional.affine(mask, angle=degree, translate=(tx, ty), scale=1, shear=0)
        return img, mask

    def __call__(self, img, mask):
        img = self.color_jitter(img)
        img, mask = self.perspective(img, mask)
        img, mask = self.translate(img, mask)
        if np.random.rand() < self.motion_blur_prob:
            img = self.motion_blur(img)
        return np.asarray(img), mask