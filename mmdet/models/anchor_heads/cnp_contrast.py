from itertools import accumulate
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, bbox_overlaps, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule, Scale, MemoryN2N
from mmdet.ops import DeformConv, CropSplit, CropSplitGt
from ..losses import cross_entropy, accuracy
import torch.nn.functional as F
import pycocotools.mask as mask_util
import numpy as np

from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions import Categorical

INF = 1e8

def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted x, y, w, h form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                      boxes[:, 2:] - boxes[:, :2]), 1)  # w, h

    
class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = DeformConv(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2,
                                        deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, out_channels)

    def init_weights(self, bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.01)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, x_off):
        if x_off.size(2)<self.conv_adaption.kernel_size[0] or x_off.size(3)<self.conv_adaption.kernel_size[1]:
            return self.relu(self.norm(x)) # skip if the shape is too small
        offset = self.conv_offset(x_off.detach())
        x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x

@HEADS.register_module
class CNPContrastHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_box=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 loss_mask=dict(type='BCELoss', use_sigmoid=True, loss_weight=1.0),
                 memory_cfg=dict(kdim=128, moving_average_rate=0.999),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(CNPContrastHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.loss_cls = build_loss(loss_cls)
        self.loss_box = build_loss(loss_box)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_mask = build_loss(loss_mask)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.fpn_strides = [8, 16, 32, 64, 128]
        self.match_coeff = [1.0, 2.0, 10]
        
        self.loss_track = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.prev_roi_feats = None
        self.prev_bboxes = None
        self.prev_det_labels = None
        self.mem = MemoryN2N(hdim=self.in_channels, kdim=memory_cfg['kdim'],
                             moving_average_rate=memory_cfg['moving_average_rate'])

        self.cls_align = FeatureAlign(2*self.feat_channels, self.feat_channels, 3)
        self.mask_align = FeatureAlign(self.feat_channels, self.feat_channels, 3)

        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(2*self.feat_channels, 8, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.track_fuse = nn.Conv2d(self.feat_channels*3, self.feat_channels, 1, padding=0)
        self.mask_fuse = nn.Conv2d(self.feat_channels*3, self.feat_channels, 1, padding=0)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation= 'relu' if i < self.stacked_convs - 2 else None,
                    bias=self.norm_cfg is None))
            
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.cls_convs = nn.Sequential(*self.cls_convs)
        self.mask_convs = nn.Sequential(*self.mask_convs)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])


        self.relu = nn.ReLU(inplace=True)
        self.crop_gt_cuda = CropSplitGt(1)
        self.crop_cuda = CropSplit(1)

        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
  
        self.track_convs = nn.ModuleList()
        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.track_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.track_convs = nn.Sequential(*self.track_convs)
        

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        for m in self.track_convs:
            normal_init(m.conv, std=0.01)


        normal_init(self.fcos_centerness, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.track_fuse, std=0.01)
        normal_init(self.mask_fuse, std=0.01)

        self.cls_align.init_weights()
        self.mask_align.init_weights()



    def forward(self, feats, feats_x, gt_bboxes=None, flag_train=True):
        cls_scores = []
        reg_maps = []
        feat_masks = []
        track_feats = []
        track_feats_ref = []
        feat_centers = []
        rel_pos_maps = []
        count = 0

        shape_h, shape_w = feats[0].shape[2:]
        unscale_factor = torch.tensor([8*shape_w, 8*shape_h, 8*shape_w, 8*shape_h], 
                                      dtype=feats[0].dtype, device=feats[0].device) #x,y,x,y
        ori_img_shape = 8*torch.tensor([shape_w, shape_h], dtype=feats[0].dtype, device=feats[0].device)
        for x, x_f, scale, stride in zip(feats, feats_x, self.scales, self.strides):
            cls_feat = x
            mask_feat = x
            track_feat = x
            track_feat_f = x_f

            cls_feat = self.cls_convs(cls_feat)
            mask_feat = self.mask_convs(mask_feat)

            points = self.get_points(x.shape[2:], stride, x.dtype, x.device)  
            rel_pos = points/ori_img_shape[None, None, :]

            if count < 3:
                track_feat = self.track_convs(track_feat)
                track_feat = F.interpolate(track_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                track_feats.append(track_feat)
                if flag_train:
                    track_feat_f = self.track_convs(track_feat_f)
                    track_feat_f = F.interpolate(track_feat_f, scale_factor=(2 ** count), mode='bilinear',
                                                    align_corners=False)
                    track_feats_ref.append(track_feat_f)


            if self.training:
                gts_ltrb, masks = multi_apply(self.get_reg_cls_target_single,
                                            gt_bboxes, points=points)
    
                gts_ltrb = torch.stack(gts_ltrb, dim=0) # batch_size, num_points, 4
                gts_ltrb = (gts_ltrb/unscale_factor[None, None, :] - 0.5)*2 #normalize to [-1,1]
                gts_ltrb = gts_ltrb.permute(0, 2, 1).view(x.size(0), 4, x.size(2), x.size(3)) # batch_size, 4, h, w
                masks = torch.cat(masks) # batch_size*num_points
                masks = masks if masks.sum() else None
                context = self.mem(cls_feat, gts_ltrb, masks, update_flag=self.training) #b,c,h,w

            else: # inference
                context = self.mem(cls_feat, update_flag=self.training) #b,c,h,w

            cls_feat = torch.cat([cls_feat, context], dim=1) #b,in_channels+256,h,w
            

            centerness = self.fcos_centerness(mask_feat)
            feat_centers.append(centerness)

            reg_map = self.fcos_reg(cls_feat) #b,8,h,w

            cls_feat = self.cls_align(cls_feat, scale(reg_map[:, :4].sigmoid() * unscale_factor[None, :, None, None]/stride))
            mask_feat = self.mask_align(mask_feat, scale(reg_map[:, :4].sigmoid() * unscale_factor[None, :, None, None]/stride))

            cls_scores.append(self.fcos_cls(cls_feat))
            reg_maps.append(reg_map)
            rel_pos_maps.append(rel_pos)

            if count < 3:
                # ################contextual enhanced##################
                feat_up = F.interpolate(mask_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                feat_masks.append(feat_up)
                count = count + 1

        # ################contextual enhanced##################
        feat_masks = torch.cat(feat_masks, dim=1)
        feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)
        feat_masks = self.relu(self.mask_fuse(feat_masks))

        track_feats = torch.cat(track_feats, dim=1)
        track_feats = self.track_fuse(track_feats)

        if flag_train:
            track_feats_ref = torch.cat(track_feats_ref, dim=1)
            track_feats_ref = self.track_fuse(track_feats_ref)
            return cls_scores, reg_maps, rel_pos_maps, feat_centers, feat_masks, track_feats, track_feats_ref
        else:
            return cls_scores, reg_maps, rel_pos_maps, feat_centers, feat_masks, track_feats, track_feats

    @force_fp32(apply_to=('feat_masks'))
    def loss(self,
             cls_scores,
             reg_maps,
             rel_pos_maps,
             feat_centers,
             feat_masks,
             track_feats,
             track_feats_ref,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             gt_masks_list=None,
             ref_bboxes_list=None,
             gt_pids_list=None):
        
        matched_gt_ids = {}
        shape_h, shape_w = feat_masks.shape[2:]
        unscale_factor = torch.tensor([shape_w, shape_h, shape_w, shape_h], 
                                      dtype=feat_masks.dtype, device=feat_masks.device) #x,y,x,y
        
        first_stage_shape = reg_maps[0].shape[2:]

        points = self.get_points(first_stage_shape, 8, reg_maps[0].dtype, reg_maps[0].device)

                
        combined_center_feats = [feat_center.flatten(2)
            for feat_center in feat_centers
            ]
    

        combined_cls_scores = [
            score.flatten(2)
            for score in cls_scores
        ]

        combined_reg_map = [
                reg_map.flatten(2)
            for reg_map in reg_maps]
        
        
        rel_pos_maps = torch.cat(rel_pos_maps, 1)
        
        combined_center_feats = torch.cat(combined_center_feats, dim=-1) #N,1,num_points
        combined_cls_scores = torch.cat(combined_cls_scores, dim=-1) #N,K,num_points
        combined_reg_map = torch.cat(combined_reg_map, dim=-1) #N,8,num_points
        
        mu = combined_reg_map[:,:4,:].transpose(1,2).sigmoid() #N, num_dt, 4 (ltrb)
        sigma = combined_reg_map[:,4:,:].transpose(1,2) #N, num_dt, 4
        scaled_dt = self.ltrb2xyxy(rel_pos_maps, mu) # N, num_dt, 4 (xyxy)
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        unscaled_dt = scaled_dt * 2*unscale_factor[None,None,:] # N, num_dt, 4

        loss_centerness = 0
        loss_cnp = 0
        loss_cls = 0
        loss_bbox = 0
        matched_num = []
        matched_ids = {}

        points_stages = [x.shape[2]*x.shape[3] for x in reg_maps]
        points_stages = list(accumulate(points_stages))


        for img_id in range(mu.size(0)):
            mu_img = scaled_dt[img_id] #num_dt, 4
            sigma_img = sigma[img_id] #num_dt, 4
            unscaled_dt_img = unscaled_dt[img_id]
            scaled_dt_img = scaled_dt[img_id]
            gts = gt_bboxes[img_id] #num_gt, 4
            if not gts.size(0):
                continue

            matched_id = [[],[]]
            tgt_uncert = None

            with torch.no_grad():
                gts_ltrb = self.xyxy2ltrb(gts, points) # num_points, num_gt, 4
                cent_ori = self.centerness_target(gts_ltrb) # num_points, 1
                cent_ori = cent_ori.view(1,1,first_stage_shape[0],first_stage_shape[1]) # 1,1,H,W

                cent_target = [cent_ori.flatten()]
                for stage in range(1, len(self.strides)):
                    cent_down = F.interpolate(cent_ori, size=reg_maps[stage].shape[2:], mode='bilinear', align_corners=False).flatten() # H/2,W/2
                    cent_target.append(cent_down)

                cent_target = torch.cat(cent_target) #total_num_points

                points_in = rel_pos_maps[0].unsqueeze(1)*(2*unscale_factor[None,:2])
                points_in = points_in.repeat(1, gts.size(0), 1)
                # Create the conditions necessary to determine if a point is within a bounding box.
                # x >= left, x <= right, y >= top, y <= bottom
                c1 = points_in[:, :, 0] <= gts[:, 2]
                c2 = points_in[:, :, 0] >= gts[:, 0]
                c3 = points_in[:, :, 1] <= gts[:, 3]
                c4 = points_in[:, :, 1] >= gts[:, 1]
                # collect points that are within a bounding box.
                mask = c1.float() + c2.float() + c3.float() + c4.float()
                mask = (mask==4).sum(dim=-1).bool()
                
                gts_cent = (gts[:,:2] + gts[:,2:])/2 /(2*unscale_factor[None,:2]) #num_gt, 2

                off_x = (rel_pos_maps[0][:,None,0] - gts_cent[None,:,0]) * unscale_factor[0] #num_dt, num_gt
                off_y = (rel_pos_maps[0][:,None,1] - gts_cent[None,:,1]) * unscale_factor[1] #num_dt, num_gt
                off_dist = torch.sqrt(off_x**2 + off_y**2) #num_dt, num_gt
                closest_offset, closest_gt_ind = off_dist.min(1) #num_dt

                pos_points, neg_points = self.assigner(closest_offset,
                                            closest_gt_ind,
                                            unscale_factor,
                                            mask,
                                            points_stages,
                                            sigma_img.mean(-1),
                                            combined_cls_scores[img_id].transpose(0,1),
                                            unscaled_dt_img,
                                            gt_bboxes[img_id],
                                            cent_target,
                                            gt_labels[img_id],
                                            mode=cfg.assigner.type)

                gt_ind = closest_gt_ind[pos_points] #num_val_dt
                dt_ind = torch.arange(off_x.size(0), device=off_x.device)[pos_points] #num_val_dt
                pair_ind = torch.cat([dt_ind[:,None], gt_ind[:,None]], dim=1)
                matched_num.append(pair_ind.size(0))

            if pair_ind.size(0) > 0:
                loss_bbox += self.loss_box(unscaled_dt_img[pair_ind[:,0]], 
                                            gts[pair_ind[:,1]],
                                            weight=cent_target[pair_ind[:,0]],
                                            avg_factor=cent_target[pair_ind[:,0]].sum())

                pos_neg_points = pos_points | neg_points
                boxcls = combined_cls_scores[img_id].transpose(0,1) #num_points, K
                boxtgt = pos_points.new_zeros(unscaled_dt_img.size(0), dtype=torch.long, requires_grad=False) #num_dt
                boxtgt[pos_points] = gt_labels[img_id][pair_ind[:,1]] #num_pos
                loss_cls += self.loss_cls(boxcls[pos_neg_points], boxtgt[pos_neg_points])/pos_points.sum()
                
                with torch.no_grad():
                    cls_uncert = (1-boxcls.sigmoid()[pos_points, boxtgt[pos_points]-1])
                    box_uncert = (1-bbox_overlaps(unscaled_dt_img[pair_ind[:,0]], gts[pair_ind[:,1]], is_aligned=True))
                    tgt_uncert = cls_uncert + box_uncert
  

            loss_cnp += self.cnp_scoreing_rule(pair_ind, 
                                                mu_img, 
                                                sigma_img, 
                                                cent_target, 
                                                tgt_uncert,
                                                gts/(2*unscale_factor[None,:]), 
                                                pos_points,
                                                mode='GMM')
 
            matched_ids[img_id] = (pair_ind[...,0], pair_ind[...,1])

            if mask.sum():
                cent_pred = combined_center_feats[img_id]
                loss_centerness += self.loss_centerness(cent_pred.flatten()[mask], cent_target.flatten()[mask])

        

        if torch.is_tensor(loss_cnp):
            loss_cnp = loss_cnp/len(matched_num)
        else:
            loss_cnp = feat_masks[0].sum() * 0

        if torch.is_tensor(loss_bbox):
            loss_bbox = loss_bbox/len(matched_num)
        else:
            loss_bbox = feat_masks[0].sum() * 0

        if torch.is_tensor(loss_cls):
            loss_cls = loss_cls/len(matched_num)
        else:
            loss_cls = feat_masks[0].sum() * 0

        if torch.is_tensor(loss_centerness):
            loss_centerness = loss_centerness/feat_masks.size(0)
        else:
            loss_centerness = feat_masks[0].sum() * 0

        gt_masks = []
        for i in range(len(gt_labels)):
            gt_label = gt_labels[i]
            gt_masks.append(
                torch.from_numpy(np.array(gt_masks_list[i][:gt_label.shape[0]], dtype=np.float32)).to(gt_label.device))
        
        loss_match = 0
        loss_seg = 0
        match_acc = 0
        n_total = 0
        mask_shape_list = []
        pos_pairs = []
        neg_vecs = []
        neg_vecs_ref = []

        for i, matched_id in matched_ids.items():#per image
            matched_dt_ids, matched_gt_ids = matched_id
            bbox_dt = unscaled_dt[i][matched_dt_ids, :]/2
            bbox_gt = gt_bboxes[i][matched_gt_ids, :]
            cur_ids = gt_pids_list[i][matched_gt_ids]

            bbox_dt = bbox_dt.detach()
            img_mask = feat_masks[i]
            mask_shape_list.append(img_mask.shape[-2:])
            mask_h = img_mask.shape[1]
            mask_w = img_mask.shape[2]

            # clamp bbox regression range
            bbox_dt[:, 0] = bbox_dt[:, 0].clamp(min=0, max=mask_w)
            bbox_dt[:, 1] = bbox_dt[:, 1].clamp(min=0, max=mask_h)
            bbox_dt[:, 2] = bbox_dt[:, 2].clamp(min=0, max=mask_w)
            bbox_dt[:, 3] = bbox_dt[:, 3].clamp(min=0, max=mask_h)

            ws = bbox_dt[:, 2] - bbox_dt[:, 0]
            hs = bbox_dt[:, 3] - bbox_dt[:, 1]
            idx_gt = torch.bitwise_and(hs>4.0, ws>4.0)
            bbox_dt = bbox_dt[idx_gt]
            cur_ids = cur_ids[idx_gt] #num_dt_boxes, 1


            if bbox_dt.shape[0] == 0:
                continue
            
            #######spp###########################
            gt_mask = gt_masks[i][matched_gt_ids][idx_gt]
            gt_mask = F.interpolate(gt_mask.unsqueeze(0), scale_factor=0.5, mode='bilinear',
                                align_corners=False).squeeze(0)

            shape = np.minimum(feat_masks[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float().contiguous() #num_dt_boxes,h,w


            ###calculate instance independent coefficient and cls_score
            bbox_gt = bbox_gt[idx_gt]
            ious = bbox_overlaps(bbox_gt/2, bbox_dt, is_aligned=True) #num_dt_boxes
            ins_labels = gt_labels[i][matched_gt_ids][idx_gt]- 1
            box_scores = combined_cls_scores[i].transpose(0,1)[matched_dt_ids][idx_gt, ins_labels].detach()
            weighting = ious * box_scores.sigmoid()
            weighting = weighting / (torch.sum(weighting) + 0.0001) * len(weighting)

            ###track feats as mask coef###
            coefs = self.extract_box_feature_center_single(track_feats[i], bbox_dt * 2) #num_dt_boxes,c

            pred_mask = torch.sigmoid(img_mask.permute(1,2,0).contiguous() @ coefs.T) #h,w,num_dt_boxes
            gt_mask_new = gt_mask_new.permute(1,2,0).contiguous() #h,w,num_dt_boxes
            pred_masks_crop = self.crop_cuda(pred_mask.unsqueeze(0), bbox_dt)
            gt_masks_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt) 

            # pred_masks_crop = self.crop_feat(pred_mask, bbox_dt) #num_dt_boxes, points
            # gt_masks_crop = self.crop_feat(gt_mask_new, bbox_dt) #num_dt_boxes, points
            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            weighted_loss_seg = F.binary_cross_entropy(pred_masks_crop, gt_masks_crop, reduction='none').sum((0,1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]

            loss_seg += (weighted_loss_seg * weighting.detach()).sum()
            ###################track####################
            bboxes = ref_bboxes_list[i]
            amplitude = 0.05
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)

            track_vec_i = coefs.T #c,q
            track_vec_i_unique = self.extract_box_feature_center_single(track_feats[i], gt_bboxes[i]).T
            track_vec_ref = self.extract_box_feature_center_single(track_feats_ref[i], new_bboxes).T #c,k

            for id, ref_id in enumerate(gt_pids_list[i]):
                if ref_id:
                    pos_pairs.append(torch.stack([track_vec_i_unique[:,id], track_vec_ref[:,ref_id-1]], dim=1)) 
               
            neg_vecs.append(track_vec_i_unique) #c,q
            neg_vecs_ref.append(track_vec_ref) #c,k

            dummy = track_vec_i.new_zeros(track_vec_i.size(1),1) #q,1
            sim_mat = F.cosine_similarity(track_vec_i[:,:,None], track_vec_ref[:,None,:], 0) #q,k
            sim_mat = torch.cat([dummy, sim_mat], dim=1) #q,k+1
            
            n_total += len(idx_gt)
            match_acc += accuracy(sim_mat, cur_ids) * len(idx_gt)


        if len(pos_pairs):
            pos_pairs = torch.stack(pos_pairs, dim=0) #Q,c,2
            sim_pos = F.cosine_similarity(pos_pairs[:,:,0], pos_pairs[:,:,1], 1) / 0.1
        else:
            sim_pos = feat_masks.new_ones(feat_masks.size(0)) / 0.1
        
        if len(neg_vecs) and len(neg_vecs_ref):
            neg_vecs = torch.cat(neg_vecs, dim=1) #c,Q
            neg_vecs_ref = torch.cat(neg_vecs_ref, dim=1) #c,K
            
            sim_intra = F.cosine_similarity(neg_vecs[:,:,None], neg_vecs[:,None,:], 0)  / 0.1 #q,q
            n = sim_intra.size(0)
            sim_intra = sim_intra.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1) #q,q-1 keep only off-diagonal
            sim_intra = sim_intra.topk(min(5*feat_masks.size(0), sim_intra.size(1)), dim=1)[0] #q, topk

            sim_inter = F.cosine_similarity(neg_vecs[:,:,None], neg_vecs_ref[:,None,:], 0) / 0.1 #q,K
            sim_inter = sim_inter.topk(min(5*feat_masks.size(0), sim_inter.size(1)), dim=1)[0] #q, topk

            sim_neg = torch.cat([sim_intra.flatten(), sim_inter.flatten()], dim=0)  

        else:
            sim_neg = feat_masks.new_ones(feat_masks.size(0)) / 0.1
        

        loss_match = -torch.log(torch.exp(sim_pos).sum()/torch.exp(sim_neg).sum())

        if torch.is_tensor(loss_seg):
            loss_seg = loss_seg/len(matched_num)
        else:
            loss_seg = feat_masks.sum() * 0

        if match_acc ==0:
            match_acc = feat_masks.sum()*0
            loss_match = feat_masks.sum()*0
        else:
            match_acc = match_acc/n_total
            loss_match = loss_match/len(matched_num)

        return dict(
            loss_seg=loss_seg,
            loss_cls=loss_cls,
            loss_cnp=loss_cnp,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_match=loss_match,
            match_acc=match_acc),\
            dict(#tensorboard log objects
            img_metas=img_metas,
            mask_shape_list=mask_shape_list,
            dt_boxes=unscaled_dt[0][matched_ids[0][0],:] if torch.any(matched_ids[0][0].bool()) else gt_bboxes[0],
            gt_bboxes=gt_bboxes[0]
            )


    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                        device=torch.cuda.current_device()) * 0
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                                     device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)

        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                   torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                   + self.match_coeff[2] * label_delta

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   reg_maps,
                   rel_pos_maps,
                   feat_centers,
                   feat_masks,
                   track_feats,
                   track_feats_ref,
                   img_metas,
                   cfg,
                   rescale=None):

        result_list = []

        shape_h, shape_w = feat_masks.shape[2], feat_masks.shape[3] 
        unscale_factor = torch.tensor([shape_w, shape_h, shape_w, shape_h], 
                                      dtype=feat_masks.dtype, device=feat_masks.device) #x,y,x,y

        combined_center_feats = [feat_center.flatten(2)
            for feat_center in feat_centers
            ]

        combined_cls_scores = [
            score.flatten(2)
            for score in cls_scores
        ]

        combined_reg_map = [
                reg_map.flatten(2)
            for reg_map in reg_maps]
        
        points_stages = [x.numel() for x in combined_center_feats]


        combined_center_feats = torch.cat(combined_center_feats, dim=-1).squeeze(1) #N,num_points
        combined_cls_scores = torch.cat(combined_cls_scores, dim=-1) #N,K,num_points
        combined_reg_map = torch.cat(combined_reg_map, dim=-1) #N,8,num_points


        rel_pos_maps = torch.cat(rel_pos_maps, 1)
        mu = combined_reg_map[:,:4,:].transpose(1,2).sigmoid() #N, num_dt, 4 (ltrb)
        sigma = combined_reg_map[:,4:,:].transpose(1,2) #N, num_dt, 4
        scaled_dt = self.ltrb2xyxy(rel_pos_maps, mu) # N, num_dt, 4
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        unscaled_dt = scaled_dt * 2*unscale_factor[None,None,:] # N, num_dt, 4


        for img_id in range(unscaled_dt.size(0)):
            track_feat_img = track_feats[img_id]
            is_first = img_metas[img_id]['is_first']

            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            det_bboxes = self.get_bboxes_single(unscaled_dt[img_id], 
                                                sigma[img_id],
                                                combined_cls_scores[img_id], 
                                                combined_center_feats[img_id], 
                                                feat_masks[img_id], 
                                                track_feat_img,
                                                points_stages,
                                                img_shape, 
                                                ori_shape,
                                                scale_factor, 
                                                cfg, 
                                                rescale)
            if det_bboxes[0].shape[0] == 0:
                cls_segms = [[] for _ in range(self.num_classes - 1)]
                result_list.append([det_bboxes[0], det_bboxes[1], cls_segms, []])
                return result_list

            res_det_bboxes = det_bboxes[0] + 0.0
            if rescale:
                res_det_bboxes[:, :4] *= scale_factor

            bbox_cent = (res_det_bboxes[:, :2] + res_det_bboxes[:, 2:4])/2 / self.strides[0] #8x downscale to match track feature
            det_roi_feats = track_feat_img[:, torch.round(bbox_cent[:,1]).long(), torch.round(bbox_cent[:,0]).long()] #c,n
            det_roi_feats = det_roi_feats.transpose(0,1) #n,c

            # recompute bbox match feature
            det_labels = det_bboxes[1]
            if is_first or (not is_first and self.prev_bboxes is None):
                det_obj_ids = np.arange(res_det_bboxes.size(0))
                # save bbox and features for later matching
                self.prev_bboxes = det_bboxes[0]
                self.prev_roi_feats = det_roi_feats
                self.prev_det_labels = det_labels
            else:

                assert self.prev_roi_feats is not None
                # only support one image at a time
                # prod = torch.mm(det_roi_feats, self.prev_roi_feats.transpose(0,1)) #n,m
                prod = F.cosine_similarity(det_roi_feats[:,None,:], self.prev_roi_feats[None,:,:], dim=-1)/0.1 #n,m
                dummy = torch.zeros(prod.size(0), 1, device=torch.cuda.current_device())
                match_score = torch.cat([dummy, prod], dim=1)
                match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
                label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
                bbox_ious = bbox_overlaps(det_bboxes[0][:, :4], self.prev_bboxes[:, :4])
                # compute comprehensive score
                comp_scores = self.compute_comp_scores(match_logprob,
                                                       det_bboxes[0][:, 4].view(-1, 1),
                                                       bbox_ious,
                                                       label_delta,
                                                       add_bbox_dummy=True)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                match_ids = match_ids.cpu().numpy().astype(np.int32)
                det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
                best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        # add new object
                        det_obj_ids[idx] = self.prev_roi_feats.size(0)
                        self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                        self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[0][idx][None]), dim=0)
                        self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                    else:
                        # multiple candidate might match with previous object, here we choose the one with
                        # largest comprehensive score
                        obj_id = match_id - 1
                        match_score = comp_scores[idx, match_id]
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            # udpate feature
                            self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                            self.prev_bboxes[obj_id] = det_bboxes[0][idx]

        obj_segms = {}
        masks = det_bboxes[2]

        for i in range(det_bboxes[0].shape[0]):
            mask = masks[i].cpu().numpy()
            im_mask = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
            shape = np.minimum(mask.shape, ori_shape[0:2])
            im_mask[:shape[0], :shape[1]] = mask[:shape[0], :shape[1]]
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            if det_obj_ids[i] >= 0:
                obj_segms[det_obj_ids[i]] = rle

        result_list.append([det_bboxes[0], det_bboxes[1], obj_segms, det_obj_ids])
        return result_list

    def get_bboxes_single(self,
                          dt_boxes, 
                          uncertainty, 
                          cls_scores,
                          feat_centers,
                          feat_mask,
                          feat_track,
                          points_stages,
                          img_shape,
                          ori_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        '''
        dt_boxes: (N,4)
        uncertainty: (N,4)
        cls_scores: (C,N)
        feat_centers: (N)
        feat_mask: (C,H//2,W//2)
        feat_track: (C,H//8,W//8)
        points_stages: (N1, N2, N3, N4)
        '''
        cls_scores = cls_scores.transpose(0,1).sigmoid() # (N,C)
        dt_boxes = dt_boxes #num_points, 4 
        centerness = feat_centers.flatten().sigmoid() #num_points
        det_bboxes = dt_boxes
        det_uncert = uncertainty.clamp(max=1) #num_points, 4

        if not det_bboxes.size(0):
            return det_bboxes, det_bboxes, det_bboxes

        ###eliminate unconfident detections before NMS###
        stage_scores = cls_scores.split(points_stages, dim=0) #num_stages, num_dt_boxes, C
        stage_cents = centerness.split(points_stages, dim=0) #num_stages, num_dt_boxes
        stage_bboxs = det_bboxes.split(points_stages, dim=0) #num_stages, num_dt_boxes, 4
        stage_uncert = det_uncert.split(points_stages, dim=0) #num_stages, num_dt_boxes, 4

        mlvl_cents = []
        mlvl_scores = []
        mlvl_bboxs = []
        mlvl_uncert = []
        for i in range(len(stage_scores)):
            cls_scores = stage_scores[i]
            centerness = stage_cents[i]
            det_bboxes = stage_bboxs[i]
            det_uncert = stage_uncert[i]

            max_scores, _ = (cls_scores * centerness[:, None]).max(dim=1)
            _, top_ind = max_scores.topk(min(cfg.max_pre_nms,cls_scores.size(0)))
            box_cents = centerness[top_ind]
            det_bboxes = det_bboxes[top_ind]
            det_uncert = det_uncert[top_ind]
            box_scores = cls_scores[top_ind]

            mlvl_bboxs.append(det_bboxes)
            mlvl_scores.append(box_scores)
            mlvl_cents.append(box_cents)
            mlvl_uncert.append(det_uncert)

        mlvl_scores = torch.cat(mlvl_scores, dim=0)
        mlvl_cents = torch.cat(mlvl_cents, dim=0)
        mlvl_bboxs = torch.cat(mlvl_bboxs, dim=0)
        mlvl_uncert = torch.cat(mlvl_uncert, dim=0)

        combined_scores = mlvl_scores * mlvl_cents[:, None] #num_dt_boxes, C
        det_bboxes, det_labels, idx = self.fast_nms(mlvl_bboxs, combined_scores.transpose(0,1), cfg)
        # each det_box is of shape [x1,y1,x2,y2,score,uncert,centerness]
        det_bboxes = torch.cat([det_bboxes, mlvl_uncert[idx]-0.1, mlvl_cents[idx].unsqueeze(1)], dim=1) 

 
        masks = []
        if det_bboxes.shape[0] > 0:
            scale = 2
            #####spp########################
            cof_pred = self.extract_box_feature_center_single(feat_track, det_bboxes[:,:4]).T #c,num_dt_boxes
            img_mask = feat_mask.permute(1, 2, 0)
            pred_mask = torch.sigmoid(img_mask.contiguous() @ cof_pred) #h,w,num_dt_boxes
            pos_masks = self.crop_cuda(pred_mask.unsqueeze(0), det_bboxes[:, :4]/scale)
            pos_masks = pos_masks.permute(2, 0, 1)
            if rescale:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale / scale_factor, mode='bilinear',
                                      align_corners=False).squeeze(0)
                det_bboxes[:,:4] /= det_bboxes.new_tensor(scale_factor)
            else:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale, mode='bilinear',
                                      align_corners=False).squeeze(0)
            masks.gt_(0.5)

        return det_bboxes, det_labels, masks 

    

    def fast_nms(self, boxes, scores, cfg):
        '''
        boxes: [num_dets,4]
        scores: [num_classes, num_dets]
        '''
        scores, idx_pre = scores.sort(1, descending=True)
        idx_pre = idx_pre[:, :cfg.max_pre_nms].contiguous()
        scores = scores[:, :cfg.max_pre_nms]

        num_classes, num_dets = idx_pre.size()

        boxes = boxes[idx_pre.view(-1), :].view(num_classes, num_dets, 4)

        iou = self.jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= cfg.nms.iou_thr)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        keep *= (scores > cfg.score_thr)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_per_img]
        scores = scores[:cfg.max_per_img]

        classes = classes[idx]
        boxes = boxes[idx]

        boxes = torch.cat([boxes, scores[:, None]], dim=1)
        return boxes, classes, idx_pre[keep][idx]

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        use_batch = True
        if box_a.dim() == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
                  (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
                  (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [n,A,4].
          box_b: (tensor) bounding boxes, Shape: [n,B,4].
        Return:
          (tensor) intersection area, Shape: [n,A,B].
        """
        n = box_a.size(0)
        A = box_a.size(1)
        B = box_b.size(1)
        max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                           box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
        min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                           box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, :, 0] * inter[:, :, :, 1]

    def centerness_target(self, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = bbox_targets[:, :, [0, 2]]
        top_bottom = bbox_targets[:, :, [1, 3]]
        left_right = left_right.clamp(min=0)
        top_bottom = top_bottom.clamp(min=0)

        centerness_targets = (
                                left_right.min(dim=-1)[0] / (left_right.max(dim=-1)[0]+1e-9)) * (
                                top_bottom.min(dim=-1)[0] / (top_bottom.max(dim=-1)[0]+1e-9))
        centerness_targets = centerness_targets.max(1)[0] 
        return torch.sqrt(centerness_targets)

    def get_points(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride / 2
        return points

    def xyxy2ltrb(self, bboxes, points):
        '''Convert bboxes from [x1, y1, x2, y2] to [l, t, r, b] format.'''
        num_points = points.size(0)
        num_gts = bboxes.size(0)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        left = xs - bboxes[..., 0]
        right = bboxes[..., 2] - xs
        top = ys - bboxes[..., 1]
        bottom = bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        bbox_targets = bbox_targets
        return bbox_targets

    def ltrb2xyxy(self, points, distance):
        """Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (n, 2), [x, y].
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom).
            max_shape (tuple): Shape of the image.

        Returns:
            Tensor: Decoded bboxes.
        """
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        x1 = x1.clamp(min=0, max=1)
        y1 = y1.clamp(min=0, max=1)
        x2 = x2.clamp(min=0, max=1)
        y2 = y2.clamp(min=0, max=1)
        return torch.stack([x1, y1, x2, y2], -1)

    def assigner(self, 
                closest_offset, 
                closest_gt_ind,
                unscale_factor, 
                mask, 
                points_stages,
                uncertainty=None,
                box_cls=None,
                box_reg=None,
                box_gts=None,
                cent_target=None,
                cof_preds=None,
                gt_label=None,
                mode='none'):
        if mode == 'none':#default mode acts similar to FCOS/sipmask wo range
            pos_points = torch.cat([(closest_offset[:points_stages[0]]<(1.5*8)),
                                    (closest_offset[points_stages[0]:points_stages[1]]<(1.5*16)),
                                    (closest_offset[points_stages[1]:points_stages[2]]<(1.5*32)),
                                    (closest_offset[points_stages[2]:]<(1.5*64))]) & mask
            neg_points = ~pos_points

        elif mode == 'uncert':
            with torch.no_grad():
                uncertainty = uncertainty.clamp(max=1)
                score = -uncertainty + cent_target
                mask_offset = score.new_zeros(score.shape).bool()
                sel_points = []
                for lvl in range(len(points_stages)):
                    start = 0 if lvl==0 else points_stages[lvl-1]
                    end = points_stages[lvl]
                    score_stage = score[start:end]
                    gt_id_stage = closest_gt_ind[start:end]
                    range_stage = torch.arange(start, end)
                    for gt_id in gt_id_stage.unique():
                        gt_points = (gt_id_stage==gt_id)
                        topk = score_stage[gt_points].topk(min(4, gt_points.sum()), largest=True)[1]
                        sel_points.append(range_stage[gt_points][topk])
                sel_points = torch.cat(sel_points)
                mask_offset[sel_points] = True
                pos_points = mask_offset & mask
                neg_points = ~pos_points & (score<0)
                
        elif mode == 'objbox':#strategy from ECCV2022 objectbox
            pos_points = (closest_offset<1.5*2/unscale_factor[None,0]) & mask 
            neg_points = ~pos_points
        
        else:
            raise NotImplementedError
        return pos_points, neg_points
    
    def cnp_scoreing_rule(self, 
                          gt_ids, 
                          mu, 
                          sigma, 
                          cent_target, 
                          tgt_uncert,
                          gts, 
                          pos_points,
                          mode='NLL'):
        loss = 0
        if mode == 'GMM':
            for gt_id in gt_ids[:,1].unique():
                gt_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[gt_ind][:,0]
                sup_mu = mu[sup_ind]
                sup_sigma = sigma[sup_ind]
                tgt_sigma = tgt_uncert[gt_ind]
                dist = Independent(Normal(sup_mu, sup_sigma), 1)#num_valid_dt
                weight = cent_target[sup_ind]+1e-9 #num_valid_dt
                gmm = MixtureSameFamily(Categorical(weight), dist)
                gt_mu = gts[gt_id]
                loss += -gmm.log_prob(gt_mu).mean()/pos_points.numel() + F.smooth_l1_loss(sup_sigma, tgt_sigma[:,None])
            

        elif mode == 'NLL':
            for gt_id in gt_ids[:,1].unique():
                gt_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[gt_ind][:,0]
                sup_mu = mu[sup_ind]
                sup_sigma = sigma[sup_ind]-0.1
                tgt_sigma = tgt_uncert[gt_ind]
                dist = Independent(Normal(sup_mu, sup_sigma), 1)#num_valid_dt
                gt_mu = gts[gt_id]
                loss += -dist.log_prob(gt_mu).mean()/pos_points.numel() + F.smooth_l1_loss(sup_sigma, tgt_sigma[:,None])

        elif mode == 'ES':
            for gt_id in gt_ids[:,1].unique():
                gt_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[gt_ind][:,0]
                sup_mu = mu[sup_ind]
                sup_sigma = sigma[sup_ind]-0.1
                tgt_sigma = tgt_uncert[gt_ind]
                gt_mu = gts[gt_id]
                dist = Independent(Normal(sup_mu, sup_sigma), 1)
                samples = dist.sample(torch.Size([128])).mean(1) # (averaging batch dim) k samples, evnet_dim
                es = ((samples-gt_mu[None,:]).abs().mean() - 0.5*(samples[:-1,:]-samples[1:,:]).abs().mean())/pos_points.numel() 
                reg = F.smooth_l1_loss(sup_sigma, tgt_sigma[:,None])
                loss += (es + reg)


        elif mode == 'contrast':
            for gt_id in gt_ids[:,1].unique():
                sup_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[sup_ind][:,0]

                dist = Independent(Normal(mu, sigma), 1)
                log_prob = dist.log_prob(gts[gt_id].unsqueeze(0)) # dt_num
                loss += -(log_prob[sup_ind].sum())/(log_prob[pos_points].sum()+1e-9).mean()
                loss /= gt_ids[:,1].numel()


        else:
            raise NotImplementedError

        return loss
    
    def get_reg_cls_target_single(self, gt_bbox, points):
        with torch.no_grad():
            gt_ltrb = self.xyxy2ltrb(gt_bbox, points) # num_points, num_gt, 4
            areas = (gt_ltrb[..., 2] + gt_ltrb[..., 0]) * (gt_ltrb[..., 3] + gt_ltrb[..., 1]) # num_points
            inside_gt_bbox = gt_ltrb.min(-1)[0] > 0 #num_points, num_gt
            areas[inside_gt_bbox == 0] = -INF
            min_area, min_area_inds = areas.min(dim=1)
            mask = min_area > 1
            ltrb_unique = gt_ltrb[range(gt_ltrb.size(0)), min_area_inds] # num_points, 4
            # x1y1 = points - ltrb_unique[..., :2] # num_points, 2
            # x2y2 = points + ltrb_unique[..., 2:] # num_points, 2
            # gt_xyxy = torch.cat([x1y1, x2y2], dim=-1) # num_points, 4
            return ltrb_unique, mask
    
    
    def extract_box_feature_center_single(self, track_feats, gt_bboxs):
        track_box_feats = track_feats.new_zeros(gt_bboxs.size()[0], track_feats.size(0))

        #####extract feature box############
        ref_feat_stride = 8
        gt_center_xs = torch.floor((gt_bboxs[:, 2] + gt_bboxs[:, 0]) / 2.0 / ref_feat_stride).long()
        gt_center_ys = torch.floor((gt_bboxs[:, 3] + gt_bboxs[:, 1]) / 2.0 / ref_feat_stride).long()

        aa = track_feats.permute(1, 2, 0)

        #avoid cuda side error when gt_center is out of track_feats
        gt_center_xs = torch.clamp(gt_center_xs, max=aa.shape[1]-1)
        gt_center_ys = torch.clamp(gt_center_ys, max=aa.shape[0]-1)
        
        bb = aa[gt_center_ys, gt_center_xs, :]
        track_box_feats += bb

        return track_box_feats
