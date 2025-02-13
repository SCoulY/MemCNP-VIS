import torch.nn as nn

from .base import BaseDetector
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2result, bbox2result_with_id
import torch.utils.checkpoint as checkpoint

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 checkpoint=False):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.checkpoint = checkpoint
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      ref_img, # images of reference frame
                      ref_bboxes, # gt bbox of reference frame
                      gt_pids, # gt ids of current frame bbox mapped to reference frame
                      gt_masks=None,
                      proposals=None):
        if not self.checkpoint:
            x = self.extract_feat(img)
            x_f = self.extract_feat(ref_img)
        else:
            x = checkpoint.checkpoint(self.extract_feat, img)
            x_f = checkpoint.checkpoint(self.extract_feat, ref_img)
        
        outs = self.bbox_head(x, x_f)

        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses, tf_logs = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, gt_masks_list=gt_masks, ref_bboxes_list = ref_bboxes, gt_pids_list=gt_pids)
        tf_logs['img']=img[0].detach().cpu() #the first img
        tf_logs['gt_bboxes']=gt_bboxes[0].detach().cpu() #all gt_bboxes of the first img
        # if type(ret[0]) == list:
        #     tf_logs['feats']=feats #the last feature map of swin encoder (wH,wW,c)
        #     tf_logs['inds']=inds #the last indices of salient blocks in swin encoder (wH, wW, nH, k)
        return losses, tf_logs

    def simple_test(self, 
                    img, 
                    img_meta, 
                    ref_imgs,
                    ref_img_metas, 
                    proposals=None, 
                    rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, x, False)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result_with_id(det_bboxes, det_labels, det_obj_ids, self.bbox_head.num_classes)
            for det_bboxes, det_labels, cls_segms, det_obj_ids in bbox_list
        ]
        segm_results = [
            cls_segms
            for det_bboxes, det_labels, cls_segms, det_obj_ids in bbox_list
        ]
        return bbox_results[0], segm_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
