from itertools import accumulate
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, bbox_overlaps, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule, Scale, MemoryN2N
from mmdet.ops import DeformConv, CropSplit, CropSplitGt, RoIAlign
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
        self.norm = nn.GroupNorm(32, in_channels)

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
class SipMaskMemCNPHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_box=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 memory_cfg=dict(kdim=128, moving_average_rate=0.999),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(SipMaskMemCNPHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_box = build_loss(loss_box)
        self.loss_centerness = build_loss(loss_centerness)

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.match_coeff = [1.0, 2.0, 10]
        
        self.loss_track = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.prev_roi_feats = None
        self.prev_bboxes = None
        self.prev_det_labels = None
        self.mem = MemoryN2N(hdim=feat_channels, kdim=memory_cfg['kdim'], 
                             moving_average_rate=memory_cfg['moving_average_rate'])

        self.nc = 32
        self.feat_align = FeatureAlign(self.feat_channels, self.feat_channels, 3)
        self.sip_cof = nn.Conv2d(self.feat_channels, self.nc * 4, 3, padding=1)
        # self.RoIAlign = RoIAlign(out_size=(3, 3), spatial_scale=1/8, sample_num=0)
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(2*self.feat_channels+3, 4, 3, padding=1)
        # self.roi_fc = nn.Linear(self.feat_channels*3*3, self.cls_out_channels)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
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
                    bias=self.norm_cfg is None))
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])


        self.sip_mask_lat = nn.Conv2d(512, self.nc, 3, padding=1)
        self.sip_mask_lat0 = nn.Conv2d(768, 512, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.crop_cuda = CropSplit(2)
        self.crop_gt_cuda = CropSplitGt(2)
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
        self.sipmask_track = nn.Conv2d(self.feat_channels * 3, 512, 1, padding=0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.sip_cof, std=0.01)
        normal_init(self.sip_mask_lat, std=0.01)
        normal_init(self.sip_mask_lat0, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        # normal_init(self.roi_fc, std=0.01)
        normal_init(self.fcos_reg, std=0.01)
        self.feat_align.init_weights()

        for m in self.track_convs:
            normal_init(m.conv, std=0.01)

    def forward(self, feats, feats_x, flag_train=True):
        # return multi_apply(self.forward_single, feats, self.scales)
        cls_scores = []
        reg_maps = []
        feat_masks = []
        track_feats = []
        track_feats_ref = []
        feat_centers = []
        cof_preds = []
        rel_pos_maps = []
        count = 0

        for x, x_f, scale, stride in zip(feats, feats_x, self.scales, self.strides):
            cls_feat = x
            reg_feat = x
            track_feat = x
            track_feat_f = x_f

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if count < 3:
                for track_layer in self.track_convs:
                    track_feat = track_layer(track_feat)
                track_feat = F.interpolate(track_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                track_feats.append(track_feat)
                if flag_train:
                    for track_layer in self.track_convs:
                        track_feat_f = track_layer(track_feat_f)
                    track_feat_f = F.interpolate(track_feat_f, scale_factor=(2 ** count), mode='bilinear',
                                                 align_corners=False)
                    track_feats_ref.append(track_feat_f)


            context, _ = self.mem(reg_feat, update_flag=self.training) #b,c,h,w
            
            
            centerness = self.fcos_centerness(reg_feat)
            feat_centers.append(centerness)
            
            rel_pos = self.get_points(context.shape[2:], 2**(3+count), context.dtype, context.device) #b,2,h,w
            ori_img_size = 8*torch.tensor([feats[0].shape[3], feats[0].shape[2]], device=context.device)
            rel_pos = rel_pos/(ori_img_size[None, None,:])
            
            
            rel_pos_expand = rel_pos.transpose(1,2).view(1, 2, context.size(2), context.size(3)).repeat(context.size(0), 1, 1, 1)
            context = torch.cat([context, rel_pos_expand, centerness], dim=1) #b,c+3,h,w
            reg_map = self.fcos_reg(context) #b,4,h,w
            # cls_feat = self.feat_align(cls_feat, scale(reg_map[:,:4]*unscale_factor[None,:,None,None]))
            cls_scores.append(self.fcos_cls(cls_feat))
            reg_maps.append(reg_map)
            rel_pos_maps.append(rel_pos)
            ########COFFECIENTS###############
            cof_pred = self.sip_cof(cls_feat)
            cof_preds.append(cof_pred)

            # ################contextual enhanced##################
            if count < 3:
                feat_up = F.interpolate(reg_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                feat_masks.append(feat_up)
                count = count + 1

        # ################contextual enhanced##################
        feat_masks = torch.cat(feat_masks, dim=1)
        feat_masks = self.relu(self.sip_mask_lat(self.relu(self.sip_mask_lat0(feat_masks))))
        feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)

        track_feats = torch.cat(track_feats, dim=1)
        track_feats = self.sipmask_track(track_feats)

        if flag_train:
            track_feats_ref = torch.cat(track_feats_ref, dim=1)
            track_feats_ref = self.sipmask_track(track_feats_ref)
            return cls_scores, reg_maps, rel_pos_maps, cof_preds, feat_centers, feat_masks, track_feats, track_feats_ref
        else:
            return cls_scores, reg_maps, rel_pos_maps, cof_preds, feat_centers, feat_masks, track_feats, track_feats

    @force_fp32(apply_to=('feat_masks'))
    def loss(self,
             cls_scores,
             reg_maps,
             rel_pos_maps,
             cof_preds,
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

        first_stage_shape = cof_preds[0].shape[2:]

        points = self.get_points(first_stage_shape, 8, cof_preds[0].dtype, cof_preds[0].device)

                
        combined_center_feats = [feat_center.flatten(2)
            for feat_center in feat_centers
            ]

        combined_cof_preds = [
            cof_pred.flatten(2)
            for cof_pred in cof_preds
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
        combined_cof_preds = torch.cat(combined_cof_preds, dim=-1) #N,C,num_points
        combined_cls_scores = torch.cat(combined_cls_scores, dim=-1) #N,K,num_points
        combined_reg_map = torch.cat(combined_reg_map, dim=-1) #N,4,num_points
        
        mu = combined_reg_map.transpose(1,2).sigmoid() #N, num_dt, 4 (ltrb)
        # sigma = 1-combined_cls_scores.sigmoid().max(1)[0]/combined_cls_scores.sigmoid().sum(1) #N, num_dt
        # sigma = 0.1 + 0.9 * sigma

        # sigma = sigma[:,:,None].repeat(1,1,4) #N, num_dt, 4

        scaled_dt = self.ltrb2xyxy(rel_pos_maps, mu) # N, num_dt, 4
        # sigma = 0.1 + 0.9 * F.softplus(sigma)
        unscaled_dt = scaled_dt * 2*unscale_factor[None,None,:] # N, num_dt, 4
        ious = []
        for i in range(unscaled_dt.size(0)): 
            ious.append(bbox_overlaps(unscaled_dt[i], gt_bboxes[i], is_aligned=False).max(-1)[0]) #num_dt
        ious = torch.stack(ious, dim=0)
        sigma = 1-ious
        sigma = sigma[:,:,None].repeat(1,1,4) #N, num_dt, 4

        loss_centerness = 0
        loss_cnp = 0
        loss_cls = 0
        loss_bbox = 0
        matched_num = []
        matched_ids = {}

        points_stages = [cof.shape[2]*cof.shape[3] for cof in cof_preds]
        points_stages = list(accumulate(points_stages))

        for img_id in range(mu.size(0)):
            mu_img = mu[img_id] #num_dt, 4
            sigma_img = sigma[img_id] #num_dt, 4
            cent_map_img = combined_center_feats[img_id].sigmoid().squeeze(0)
            unscaled_dt_img = unscaled_dt[img_id]
            # scaled_dt_img = scaled_dt[img_id]
            gts = gt_bboxes[img_id] #num_gt, 4
            if not gts.size(0):
                continue

            matched_id = [[],[]]
            cent_target = combined_center_feats.new_zeros(combined_center_feats[img_id].size(), requires_grad=False)
            with torch.no_grad():
                ltrb_target = self.xyxy2ltrb(gts, points) # num_points, num_gt, 4
                cent_ori = self.centerness_target(ltrb_target) # num_points, 1
                cent_ori = cent_ori.view(1,1,first_stage_shape[0],first_stage_shape[1]) # 1,1,H,W
                cent_2x_down = F.interpolate(cent_ori, size=cof_preds[1].shape[2:], mode='bilinear', align_corners=False).flatten() # H/2,W/2
                cent_4x_down = F.interpolate(cent_ori, size=cof_preds[2].shape[2:], mode='bilinear', align_corners=False).flatten() # H/4,W/4
                cent_8x_down = F.interpolate(cent_ori, size=cof_preds[3].shape[2:], mode='bilinear', align_corners=False).flatten() # H/8,W/8
                cent_target = torch.cat([cent_ori.flatten(), cent_2x_down, cent_4x_down, cent_8x_down]) #total_num_points

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
                # mask = torch.from_numpy(gt_masks_list[img_id]).to(cent_target.device).float() # num_semantic_channels, H, W
                # mask1 = F.interpolate(mask.unsqueeze(0), size=cof_preds[0].shape[2:], mode='bilinear').squeeze(0).sum(0).flatten()
                # mask2 = F.interpolate(mask.unsqueeze(0), size=cof_preds[1].shape[2:], mode='bilinear').squeeze(0).sum(0).flatten()
                # mask3 = F.interpolate(mask.unsqueeze(0), size=cof_preds[2].shape[2:], mode='bilinear').squeeze(0).sum(0).flatten()
                # mask4 = F.interpolate(mask.unsqueeze(0), size=cof_preds[3].shape[2:], mode='bilinear').squeeze(0).sum(0).flatten()
                # mask = torch.cat([mask1, mask2, mask3, mask4]) >= 1 #total_num_points
                
                gts_cent = (gts[:,:2] + gts[:,2:])/2 /(2*unscale_factor[None,:2]) #num_gt, 2
                gt_l = (gts[:, 2]-gts[:, 0])/(2*unscale_factor[None,0]) #num_gt
                gt_t = (gts[:, 3]-gts[:, 1])/(2*unscale_factor[None,1]) #num_gt
                # pos_map_unnorm = rel_pos_maps[0] * 2*unscale_factor[None,:2]  #num_dt, 2
                off_l = (rel_pos_maps[0][:,None,0] - gts_cent[None,:,0]) #num_dt, num_gt
                off_t = (rel_pos_maps[0][:,None,1] - gts_cent[None,:,1]) #num_dt, num_gt
                off_dist = torch.sqrt(off_l**2 + off_t**2) #num_dt, num_gt
                closest_offset, closest_gt_ind = off_dist.min(1) #num_dt

                pos_points, neg_points = self.assigner(closest_offset,
                                            closest_gt_ind,
                                            unscale_factor,
                                            mask,
                                            points_stages,
                                            combined_cls_scores[img_id].transpose(0,1),
                                            cent_target,
                                            cof_preds,
                                            gt_labels[img_id],
                                            gt_l,
                                            gt_t,
                                            sigma_img,
                                            mode='uncertainty')

                gt_ind = closest_gt_ind[pos_points] #num_val_dt
                dt_ind = torch.arange(off_l.size(0), device=off_l.device)[pos_points] #num_val_dt
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
                loss_cls += self.loss_cls(boxcls[pos_neg_points], boxtgt[pos_neg_points])


            loss_cnp += self.cnp_scoreing_rule(pair_ind, 
                                                mu_img, 
                                                sigma_img, 
                                                cent_target, 
                                                gts/(2*unscale_factor[None,:]), 
                                                mask,
                                                mode='GMM')
 
            # closest_pair_ind = [(off_dist).min(0)[1], torch.arange(off_l.size(1), device=off_l.device)]
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
            loss_cls = loss_cls/sum(matched_num)
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

        for i, matched_id in matched_ids.items():#per image
            matched_dt_ids, matched_gt_ids = matched_id
            bbox_dt = unscaled_dt[i][matched_dt_ids, :] / 2
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
            matched_gt_masks = gt_masks[i][matched_gt_ids][idx_gt]
            gt_mask = F.interpolate(matched_gt_masks.unsqueeze(0), scale_factor=0.5, mode='bilinear',
                                align_corners=False).squeeze(0)

            shape = np.minimum(feat_masks[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()

            gt_mask_new = gt_mask_new.permute(1, 2, 0).contiguous()
            img_mask1 = img_mask.permute(1, 2, 0)
            ###calculate instance independent coefficient and cls_score
            bbox_gt = bbox_gt[idx_gt]
            ious = bbox_overlaps(bbox_gt/2, bbox_dt, is_aligned=True) #num_dt_boxes
            ins_labels = gt_labels[i][matched_gt_ids][idx_gt]- 1
            box_scores = combined_cls_scores[i].transpose(0,1)[matched_dt_ids][idx_gt, ins_labels]
            weighting = ious * box_scores.sigmoid()
            weighting = weighting / (torch.sum(weighting) + 0.0001) * len(weighting)
            # cof_pred = [combined_cof_preds[i][:, ((box[1]+box[3])/2).round().long(), ((box[0]+box[2])/2).round().long()] 
            #             for box in bbox_dt/4]
            # cof_pred = torch.stack(cof_pred, dim=0) #num_dt_boxes, C

            cof_pred = combined_cof_preds[i].transpose(0,1) 
            cof_pred = cof_pred[matched_dt_ids][idx_gt] #num_dt_boxes, C

            pos_masks00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ cof_pred[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ cof_pred[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ cof_pred[:, 96:128].t())
            pred_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)
            pred_masks = self.crop_cuda(pred_masks, bbox_dt)
            gt_mask_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt)
            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')
            
            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_seg += torch.sum(pre_loss * weighting.detach())

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

            track_feat_i = self.extract_box_feature_center_single(track_feats[i], bbox_dt * 2)
            track_box_ref = self.extract_box_feature_center_single(track_feats_ref[i], new_bboxes)

            prod = torch.mm(track_feat_i, torch.transpose(track_box_ref, 0, 1))
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())

            prod_ext = torch.cat([dummy, prod], dim=1)
            loss_match += cross_entropy(prod_ext, cur_ids)
            n_total += len(idx_gt)
            match_acc += accuracy(prod_ext, cur_ids) * len(idx_gt)

        if torch.is_tensor(loss_seg):
            loss_seg = loss_seg/len(matched_num)
        else:
            loss_seg = feat_masks[0].sum() * 0

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
                   cof_preds,
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

        first_stage_shape = (shape_h//4, shape_w//4)

        combined_center_feats = [feat_center.flatten(2)
            for feat_center in feat_centers
            ]

        combined_cof_preds = [
            cof_pred.flatten(2)
            for cof_pred in cof_preds
        ]

        combined_cls_scores = [
            score.flatten(2)
            for score in cls_scores
        ]

        combined_reg_map = [
                reg_map.flatten(2)
            for reg_map in reg_maps]


        combined_center_feats = torch.cat(combined_center_feats, dim=-1).squeeze(1) #N,num_points
        combined_cof_preds = torch.cat(combined_cof_preds, dim=-1) #N,C,num_points
        combined_cls_scores = torch.cat(combined_cls_scores, dim=-1) #N,K,num_points
        combined_reg_map = torch.cat(combined_reg_map, dim=-1) #N,8,num_points


        rel_pos_maps = torch.cat(rel_pos_maps, 1)
        mu = combined_reg_map[:,:4,:].transpose(1,2).sigmoid() #N, num_dt, 4 (ltrb)
        sigma = combined_reg_map[:,4:,:].transpose(1,2) #N, num_dt, 4
        scaled_dt = self.ltrb2xyxy(rel_pos_maps, mu) # N, num_dt, 4
        sigma = 0.1 + 0.9 * F.softplus(sigma)
        unscaled_dt = scaled_dt * 2*unscale_factor[None,None,:] # N, num_dt, 4


        for img_id in range(unscaled_dt.size(0)):
            track_feat_list = track_feats[img_id]
            is_first = img_metas[img_id]['is_first']

            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            det_bboxes = self.get_bboxes_single(unscaled_dt[img_id], 
                                                sigma[img_id],
                                                combined_cls_scores[img_id], 
                                                combined_center_feats[img_id], 
                                                feat_masks[img_id], 
                                                combined_cof_preds[img_id],
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

            det_roi_feats = self.extract_box_feature_center_single(track_feat_list, res_det_bboxes[:, :4])

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
                prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
                m = prod.size(0)
                dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
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
                          cof_preds,
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
        cof_preds: (nc, N)
        '''
        cls_scores = cls_scores.transpose(0,1).sigmoid() # (N,C)
        dt_boxes = dt_boxes #num_points, 4 
        uncertainty = uncertainty.mean(-1) #num_points
        centerness = feat_centers.flatten().sigmoid() #num_points
        # ct1 = uncertainty < uncertainty.mean() #num_points criterion 1 to filter out points
        # ct1 = cls_scores.max(1)[0] > 0.5 #num_points criterion 1 to filter out points
        # ct2 = centerness > 0.5 #num_points criterion 2 to filter out points
        # import pdb; pdb.set_trace()
        # winning_ind = ct2 #num_points
        # uncertainty = uncertainty[winning_ind] #num_points
        # dt_boxes = dt_boxes[winning_ind] #num_points, 4
        # centerness = centerness[winning_ind] #num_points
        # cls_scores = cls_scores[winning_ind] #num_points, C

        det_bboxes = dt_boxes
        det_uncert = uncertainty

        if not det_bboxes.size(0):
            return det_bboxes, det_bboxes, det_bboxes

        ###eliminate unconfident detections before NMS###
        max_scores = (cls_scores * centerness[:, None]).max(1)[0] #num_dt_boxes
        true_dets = (max_scores).sort(descending=True)[1][:cfg.max_pre_nms]
        # true_dets = true_dets[box_cents[true_dets]>0.5]
        box_cents = centerness[true_dets]
        # det_labels = det_labels[true_dets] 
        det_bboxes = det_bboxes[true_dets]
        # max_score = max_score[true_dets]
        det_uncert = det_uncert[true_dets]
        box_scores = cls_scores[true_dets]

        ###calculate instance independent coefficient, cls_score and center_score
        cof_pred = cof_preds.transpose(0,1)
        cof_pred = cof_pred[true_dets] #num_dt_boxes, C


        combined_scores = box_scores * box_cents[:, None] #num_dt_boxes, C
        det_bboxes, det_labels, cof_pred, idx = self.fast_nms(det_bboxes, combined_scores.transpose(1, 0), cof_pred, cfg)
        # each det_box is of shape [x1,y1,x2,y2,score,uncert,centerness]
        det_bboxes = torch.cat([det_bboxes, det_uncert[idx].unsqueeze(1), box_cents[idx].unsqueeze(1)], dim=1) 

        
        masks = []
        if det_bboxes.shape[0] > 0:
            scale = 2
            #####spp########################
            img_mask1 = feat_mask.permute(1, 2, 0)
            pos_masks00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ cof_pred[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ cof_pred[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ cof_pred[:, 96:128].t())
            pos_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)

            pos_masks = self.crop_cuda(pos_masks, det_bboxes[:, :4]/scale)
 
            pos_masks = pos_masks.permute(2, 0, 1)
            if rescale:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale / scale_factor, mode='bilinear',
                                      align_corners=False).squeeze(0)
                det_bboxes[:,:4] /= det_bboxes.new_tensor(scale_factor)
            else:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale, mode='bilinear',
                                      align_corners=False).squeeze(0)
            masks.gt_(0.5)

        # if det_bboxes.size(0):
        #     masks_iou = self.mask_iou(masks, masks)
        #     masks_iou.triu_(diagonal=1)
        #     iou_mask, _ = torch.max(masks_iou, dim=1)
        #     keep = (iou_mask <= 0.1) #* (boxes[:,4] > cfg.score_thr)
        #     det_bboxes = det_bboxes[keep]
        #     det_labels = det_labels[keep]
        #     masks = masks[keep]
        return det_bboxes, det_labels, masks 

    def extract_box_feature_center_single(self, track_feats, gt_bboxs):

        track_box_feats = track_feats.new_zeros(gt_bboxs.size()[0], 512)

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


    def uncert_nms(self, boxes, masks, cls_scores, box_cents, cfg):
        '''
        boxes: [num_dets,6]; 6=[x1,y1,x2,y2,score,uncert]
        masks: [num_dets,h,w]
        cls_scores: [num_dets, num_classes-1]
        box_cents: [num_dets]
        '''
        # comprehend_score = self.comprehensive_uncert_score(boxes, masks, cls_scores, box_cents, cfg)
        # comprehend_score, idx = comprehend_score.sort()
        # idx = idx[:min(cfg.max_pre_nms, idx.size(0))].contiguous()
        # comprehend_score = comprehend_score[:min(cfg.max_pre_nms, idx.size(0))]
        # boxes = boxes[idx]
        # masks = masks[idx]
        # cls_scores = cls_scores[idx]

        # min_bbox = []
        # ids = []
        # for id, mask in enumerate(masks):
        #     ind = mask.nonzero()
        #     if torch.any(ind):
        #         y1, x1 = max((ind[:,0].min()-1), 0), max((ind[:,1].min()-1), 0)
        #         y2, x2 = min((ind[:,0].max()+1), mask.shape[0]-1), min((ind[:,1].max()+1), mask.shape[1]-1)
        #         min_bbox.append(torch.tensor([x1, y1, x2, y2], device=mask.device))
        #         ids.append(id)
        # min_bbox = torch.stack(min_bbox, dim=0)

        # boxes = boxes[ids]
        # masks = masks[ids]
        # cls_scores = cls_scores[ids]
        # boxes[:,:4] = min_bbox
        
        masks_iou = self.mask_iou(masks, masks)
        masks_iou.triu_(diagonal=1)
        iou_mask, _ = torch.max(masks_iou, dim=1)

        # iou = self.jaccard(boxes[:,:4], boxes[:,:4])
        # iou.triu_(diagonal=1)
        # iou_box, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_mask <= 0.1) #* (boxes[:,4] > cfg.score_thr)

        masks = masks[keep]
        boxes = boxes[keep]
        cls_scores = cls_scores[keep]

        masks = masks[:cfg.max_per_img]
        boxes = boxes[:cfg.max_per_img]
        cls_scores = cls_scores[:cfg.max_per_img]

        classes = torch.argmax(cls_scores, dim=1) 

        return boxes, classes, masks
    
    def comprehensive_uncert_score(self, boxes, masks, cls_scores, box_cents, cfg):
        '''
        this fun. is used to compute the comprehensive uncertainty score using
        ratio of mask/box are, cls_score and uncertainty score
        '''
        # compute the mask/box area ratio
        mask_area = masks.sum(dim=(1,2))
        box_area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        mask_box_area_ratio = mask_area/box_area
        # compute the cls_score
        cls_score = torch.max(cls_scores, dim=1)[0]
        uncert = boxes[:,5]
        # compute the comprehensive uncertainty score
        comp_uncert_score = (1-cls_score) + (1-box_cents) + (1-mask_box_area_ratio)
        return comp_uncert_score

    def mask_iou(self, masks1, masks2):
        '''
        masks1: [num_dets1,h,w]
        masks2: [num_dets2,h,w]
        masks_iou: [num_dets1,num_dets2]
        '''
        masks1 = masks1.unsqueeze(1).expand(-1, masks2.size(0), -1, -1).flatten(-2, -1).bool()
        masks2 = masks2.unsqueeze(0).expand(masks1.size(0), -1, -1, -1).flatten(-2, -1).bool()
        inter = (masks1 & masks2).float().sum(-1)
        union = (masks1 | masks2).float().sum(-1)
        masks_iou =  inter / torch.clamp(union, min=1e-6)
        return masks_iou

    def fast_nms(self, boxes, scores, coefs, cfg):
        '''
        boxes: [num_dets,4]
        scores: [num_classes, num_dets]
        coefs: [num_dets, nc]; 
        '''
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :cfg.max_pre_nms].contiguous()
        scores = scores[:, :cfg.max_pre_nms]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        coefs = coefs[idx.view(-1), :].view(num_classes, num_dets, -1)

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
        coefs = coefs[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_per_img]
        scores = scores[:cfg.max_per_img]

        classes = classes[idx]
        boxes = boxes[idx]
        coefs = coefs[idx]
        boxes = torch.cat([boxes, scores[:, None]], dim=1)
        return boxes, classes, coefs, idx

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
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
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
                box_cls=None,
                cent_target=None,
                cof_preds=None,
                gt_label=None,
                gt_l=None,
                gt_t=None,
                sigma=None,
                mode='none'):
        with torch.no_grad():
            if mode == 'none':#default mode acts same as sipmask using regress range
                offset_unnorm = closest_offset * 2*unscale_factor[None,0]
                pos_points = torch.cat([(offset_unnorm[:points_stages[0]]<=100) & (closest_offset[:points_stages[0]]<(1.5*2/unscale_factor[None,0])),
                                (offset_unnorm[points_stages[0]:points_stages[1]]>100) & (offset_unnorm[points_stages[0]:points_stages[1]]<=200) & (closest_offset[points_stages[0]:points_stages[1]]<(2*1.5*2/unscale_factor[None,0])),
                                (offset_unnorm[points_stages[1]:points_stages[2]]>200) & (offset_unnorm[points_stages[1]:points_stages[2]]<=400) & (closest_offset[points_stages[1]:points_stages[2]]<(4*1.5*2/unscale_factor[None,0])),
                                (offset_unnorm[points_stages[2]:]>400) & (closest_offset[points_stages[2]:]<(8*1.5*2/unscale_factor[None,0]))]) & mask
                neg_points = ~pos_points
                
            elif mode == 'objbox':#strategy from ECCV2022 objectbox
                pos_points = (closest_offset<1.5*2/unscale_factor[None,0]) & mask 
                neg_points = ~pos_points

            elif mode == 'dynamic' and box_cls is not None and cent_target is not None and cof_preds is not None and gt_label is not None:
                score = box_cls * cent_target[:,None]
                score_0 = score[:points_stages[0]] #num_points, K 
                score_1 = score[points_stages[0]:points_stages[1]]  #num_points, K
                score_2 = score[points_stages[1]:points_stages[2]]  #num_points, K
                score_3 = score[points_stages[2]:]  #num_points, K
                score_0 = score_0.view(cof_preds[0].shape[2], cof_preds[0].shape[3], -1) #H, W, K
                score_1 = F.interpolate(score_1.transpose(0,1).unsqueeze(0).view(1, -1, cof_preds[1].shape[2], cof_preds[1].shape[3]), size=(cof_preds[0].shape[2], cof_preds[0].shape[3]), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0) #H, W, K
                score_2 = F.interpolate(score_2.transpose(0,1).unsqueeze(0).view(1, -1, cof_preds[2].shape[2], cof_preds[2].shape[3]), size=(cof_preds[0].shape[2], cof_preds[0].shape[3]), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)  #H, W, K
                score_3 = F.interpolate(score_3.transpose(0,1).unsqueeze(0).view(1, -1, cof_preds[3].shape[2], cof_preds[3].shape[3]), size=(cof_preds[0].shape[2], cof_preds[0].shape[3]), mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)  #H, W, K
                
                gt_label =  gt_label[closest_gt_ind[:points_stages[0]].view(-1,1)]-1
                score_0 = score_0.flatten(0,1).gather(1, gt_label).squeeze(1) #points stage_0
                score_1 = score_1.flatten(0,1).gather(1, gt_label).squeeze(1) #points stage_0
                score_2 = score_2.flatten(0,1).gather(1, gt_label).squeeze(1) #points stage_0
                score_3 = score_3.flatten(0,1).gather(1, gt_label).squeeze(1) #points stage_0
                stage_assign = torch.stack([score_0, score_1, score_2, score_3], dim=1) #points, stages

                stage_mean = stage_assign.mean(1, keepdim=True)
                stage_assign = (stage_assign >= stage_mean)
                stage_assign_0 = stage_assign[:,0].flatten()
                stage_assign_1 = F.interpolate((stage_assign[:,1].view(cof_preds[0].shape[2:])[None,None,:]).float(), size=(cof_preds[1].shape[2], cof_preds[1].shape[3]), mode='nearest').bool().flatten()
                stage_assign_2 = F.interpolate((stage_assign[:,2].view(cof_preds[0].shape[2:])[None,None,:]).float(), size=(cof_preds[2].shape[2], cof_preds[2].shape[3]), mode='nearest').bool().flatten()
                stage_assign_3 = F.interpolate((stage_assign[:,3].view(cof_preds[0].shape[2:])[None,None,:]).float(), size=(cof_preds[3].shape[2], cof_preds[3].shape[3]), mode='nearest').bool().flatten()

                pos_0 = (closest_offset[:points_stages[0]]<np.sqrt(2)*2/unscale_factor[None,0]) & stage_assign_0
                pos_1 = (closest_offset[points_stages[0]:points_stages[1]]<np.sqrt(2)*2**2/unscale_factor[None,0]) & stage_assign_1
                pos_2 = (closest_offset[points_stages[1]:points_stages[2]]<np.sqrt(2)*2**3/unscale_factor[None,0]) & stage_assign_2
                pos_3 = (closest_offset[points_stages[2]:]<np.sqrt(2)*2**4/unscale_factor[None,0]) & stage_assign_3

                pos_points = torch.cat([pos_0, pos_1, pos_2, pos_3], dim=0)
                neg_points = ~pos_points
            
            elif mode == 'posneg' and gt_l is not None and gt_t is not None:
                pos_points = ((closest_offset<(gt_t[closest_gt_ind]*0.2)) & (closest_offset<(gt_l[closest_gt_ind]*0.2))) 
                neg_points = (closest_offset>(gt_t[closest_gt_ind]*0.6)) & (closest_offset>(gt_l[closest_gt_ind]*0.6)) & ~mask

            elif mode == 'uncertainty' and sigma is not None:
                offset_unnorm = closest_offset * 2*unscale_factor[None,0]
                top_k_mu = torch.topk(sigma[:,0], k=int(0.01*sigma.size(0)), dim=0, largest=False)[0][-1]
                pos_points = torch.cat([(offset_unnorm[:points_stages[0]]<=100) & (sigma[:points_stages[0],0]<top_k_mu),
                                (offset_unnorm[points_stages[0]:points_stages[1]]>100) & (offset_unnorm[points_stages[0]:points_stages[1]]<=200) & (sigma[points_stages[0]:points_stages[1],0]<top_k_mu),
                                (offset_unnorm[points_stages[1]:points_stages[2]]>200) & (offset_unnorm[points_stages[1]:points_stages[2]]<=400) & (sigma[points_stages[1]:points_stages[2],0]<top_k_mu),
                                (offset_unnorm[points_stages[2]:]>400) & (sigma[points_stages[2]:,0]<top_k_mu)]) & mask
                neg_points = ~pos_points

            else:
                raise NotImplementedError
            return pos_points, neg_points
    
    def cnp_scoreing_rule(self, 
                          gt_ids, 
                          mu, 
                          sigma, 
                          cent_target, 
                          gts, 
                          mask,
                          mode='GMM'):
        loss = 0
        if mode == 'GMM':
            for gt_id in gt_ids[:,1].unique():
                sup_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[sup_ind][:,0]
                sup_mu = mu[sup_ind]
                sup_sigma = sigma[sup_ind]
                # dist = MultivariateNormal(sup_mu, torch.diag_embed(sup_sigma))
                dist = Independent(Normal(sup_mu, sup_sigma), 1)#num_valid_dt
                weight = cent_target[sup_ind]+1e-9 #num_valid_dt
                gmm = MixtureSameFamily(Categorical(weight), dist)
                gt_mu = gts[gt_id]
                loss += -gmm.log_prob(gt_mu).mean()/mask.numel()
            

        elif mode == 'NLL':
            if gt_ids[:,1].numel():
                dist = Independent(Normal(mu[gt_ids[:,0]], sigma[gt_ids[:,0]]), 1)
                loss += -dist.log_prob(gts[gt_ids[:,1]]).mean()/gt_ids[:,1].numel()

        elif mode == 'ES':
            for gt_id in gt_ids[:,1].unique():
                sup_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[sup_ind][:,0]
                sup_mu = mu[sup_ind]
                sup_sigma = sigma[sup_ind]
                gt_mu = gts[gt_id]
                dist = Independent(Normal(sup_mu, sup_sigma), 1)
                samples = dist.sample(torch.Size([128])).mean(1) # (averaging batch dim) k samples, evnet_dim
                loss += (samples-gt_mu[None,:]).abs().mean() - 0.5*(samples[:,None,:]-samples[None,:,:]).abs().mean()
                loss /= gt_ids[:,1].numel()

        elif mode == 'contrast':
            for gt_id in gt_ids[:,1].unique():
                sup_ind = gt_ids[:,1]==gt_id
                sup_ind = gt_ids[sup_ind][:,0]
                # ood_ind = gt_ids[:,1]!=gt_id
                # ood_ind = gt_ids[ood_ind][:,0]

                dist = Independent(Normal(mu, sigma), 1)
                log_prob = dist.log_prob(gts[gt_id].unsqueeze(0)) # dt_num
                loss += -(log_prob[sup_ind].sum())/(log_prob[mask].sum()+1e-9).mean()
                loss /= gt_ids[:,1].numel()



        else:
            raise NotImplementedError

        return loss