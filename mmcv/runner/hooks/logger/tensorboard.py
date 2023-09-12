import os.path as osp
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from .base import LoggerHook
from ...utils import master_only
import torchvision

import random
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    return fig

class TensorboardLoggerHook(LoggerHook):

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir

    @master_only
    def before_run(self, runner):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'tf_logs')
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        for var in runner.log_buffer.output:
            if var in ['time', 'data_time']:
                continue
            tag = '{}/{}'.format(var, runner.mode)
            record = runner.log_buffer.output[var]
            if isinstance(record, str):
                self.writer.add_text(tag, record, runner.iter)
            else:
                self.writer.add_scalar(tag, runner.log_buffer.output[var],
                                       runner.iter)

    @master_only
    def after_run(self, runner):
        self.writer.close()


class TBImgLoggerHook(LoggerHook):
    
    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 plot_img_bbox=False,
                 plot_neg_bbox=False,
                 plot_cls_score=False,
                 plot_centerness=False,
                 plot_memory=False):
        super(TBImgLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag)
        self.log_dir = log_dir
        self.img_bbox = plot_img_bbox
        self.neg_bbox = plot_neg_bbox
        self.cls_score = plot_cls_score
        self.centerness = plot_centerness
        self.memory = plot_memory

    @master_only
    def before_run(self, runner):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorflow and tensorboardX '
                              'to use TensorboardLoggerHook.')
        else:
            if self.log_dir is None:
                self.log_dir = osp.join(runner.work_dir, 'tf_logs')
            self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        #runner.log_buffer contains keys: ['pos_inds', 'pos_decoded_bbox_preds', 'pos_decoded_target_preds', 
        #                                  'pos_centerness', 'pos_centerness_targets', 'img']

        #TODO add centerness heatmap to tensorboard
        img = runner.img_log_buffer[0]['img'] 
        img_norm_cfg = runner.img_log_buffer[1]
        mean = np.array(img_norm_cfg['mean'])
        std = np.array(img_norm_cfg['std'])

        img_shape = runner.img_log_buffer[2]['img_shape']
        h, w, _ = img_shape
        img = img.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = img.astype(np.uint8)
        # import pdb; pdb.set_trace()
        self.img_inp = img
        img = img[:h, :w, :]
        self.img_ori = img.transpose(2, 0, 1) #c,h,w


        if self.img_bbox:
            self.plot_img_bbox(runner)
        if self.cls_score:
            self.plot_cls_scores(runner)
        if self.centerness:
            self.plot_centerness(runner)
        if self.memory:
            self.plot_memory(runner)
        if self.neg_bbox:
            self.plot_neg_bbox(runner)

    @master_only
    def plot_neg_bbox(self, runner):
        tag = '{}/{}'.format('img_neg_bbox', runner.mode)
        dt_bboxes = runner.img_log_buffer[0]['neg_boxes']
        self.writer.add_image_with_boxes(tag, self.img_ori, dt_bboxes, runner.iter)

    @master_only
    def plot_memory(self, runner):
        tag = '{}/{}'.format('memory', runner.mode)
        memory = runner.img_log_buffer[0]['memory'][:,-4:] #k,4
        memory = memory.view(memory.size(0)//16, 16, -1) #k//16,16,4
        memory = memory.permute(2,0,1).unsqueeze(1).repeat(1,3,1,1) #4,3,k//16,16
        self.writer.add_images(tag, memory, runner.iter)

    @master_only
    def plot_img_bbox(self, runner):
        tag = '{}/{}'.format('img_gt_bbox', runner.mode)
        gt_bboxes = runner.img_log_buffer[0]['gt_bboxes'] 
        self.writer.add_image_with_boxes(tag, self.img_ori, gt_bboxes, runner.iter)

        tag = '{}/{}'.format('img_dt_bbox', runner.mode)
        dt_bboxes = runner.img_log_buffer[0]['dt_boxes']
        self.writer.add_image_with_boxes(tag, self.img_ori, dt_bboxes, runner.iter)
    
    @master_only
    def plot_cls_scores(self, runner):
        tag = '{}/{}'.format('cls_scores', runner.mode)
        
        cls_scores = runner.img_log_buffer[0]['combined_cls_scores'] #C,H,W
        gt_masks = runner.img_log_buffer[0]['combined_gt_masks'] #C,H,W
        self.writer.add_images(tag, torch.softmax(cls_scores, dim=0).unsqueeze(1), runner.iter)

        tag = '{}/{}'.format('gt_masks', runner.mode)
        self.writer.add_images(tag, gt_masks.unsqueeze(1), runner.iter)

    
    @master_only
    def plot_centerness(self, runner):
        ###Draft visualization of centerness
        tag = '{}/{}'.format('centerness', runner.mode)
        img_shape = runner.img_log_buffer[2]['img_shape']
        h, w, _ = img_shape
        cent_pred = (255*runner.img_log_buffer[0]['cent_pred'].unsqueeze(2).numpy()).astype(np.uint8) #H,W,1
        cent_target = (255*runner.img_log_buffer[0]['cent_target'].unsqueeze(2).numpy()).astype(np.uint8) #H,W,1
        cent_pred = cent_pred[:h, :w]
        cent_target = cent_target[:h, :w]
        heat_pred = cv2.applyColorMap(cent_pred, cv2.COLORMAP_RAINBOW)
        heat_target = cv2.applyColorMap(cent_target, cv2.COLORMAP_RAINBOW)
        
        img_inp = self.img_inp
        img_inp = cv2.resize(img_inp, (cent_pred.shape[1], cent_pred.shape[0]))
        blend_pred = cv2.addWeighted(img_inp, 0.5, heat_pred, 0.5, 0)
        blend_target = cv2.addWeighted(img_inp, 0.5, heat_target, 0.5, 0)
        cent_toshow = np.concatenate([blend_pred, blend_target], axis=1)
        self.writer.add_image(tag, cent_toshow, runner.iter, dataformats='HWC')
    

    @master_only
    def boxind2imgind(self, pos_inds, w,
                      scale_h, scale_w, 
                      pad_h, pad_w,
                      x1, y1):
        coords = []
        for value in pos_inds:
            ### TODO current rescale of pos_inds is not correct
            # value *= 16 #scale to original image size
            x_down = (value%w) 
            y_down = (value//w) 
            x = x_down*4 + x1
            y = y_down*4 + y1
            ori_x = x-pad_w /scale_w
            ori_y = y-pad_h /scale_h
            coords.append([ori_y, ori_x])
        return np.array(coords).astype(np.int)

    @master_only
    def after_run(self, runner):
        self.writer.close()