"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mmdet.core import bbox_overlaps


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_iou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the iou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_iou = cost_iou
        

    @torch.no_grad()
    def forward(self, cls_boxes, cls_targets, pred_boxes, targets):
        """ Performs the matching
        Params:
            cls_boxes: cls_boxes is a list of tensors (len(cls_boxes) = batch_size), where each tensor is of shape [num_queries, num_classes] containing the classification logits
            cls_targets: cls_targets is a list of tensors (len(cls_targets) = batch_size), where each tensor is of shape [num_target_boxes] of target class
            pred_boxes: This is a list of preds (len(pred_boxes) = batch_size), where each pred is containing:
                 "boxes": Tensor of dim [num_pred_boxes, 4] containing the pred box coordinates
            
            targets: This is a list of targets (len(targets) = batch_size), where each target is containing:
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = torch.cat([p for p in pred_boxes])

        # Also cat the target boxes
        tgt_bbox = torch.cat([g for g in targets]) # [batch_size * num_target_boxes, 4]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cls_boxes = torch.cat([c for c in cls_boxes]) # [batch_size * num_queries, num_classes]
        cls_targets = torch.cat([t for t in cls_targets])-1 # [batch_size * num_target_boxes]
        cost_cls = 1 - cls_boxes.sigmoid()[:, cls_targets]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the iou cost betwen boxes
        cost_iou = -bbox_overlaps(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_class * cost_cls + self.cost_bbox * cost_bbox  + self.cost_iou * cost_iou # bs*num_queries, bs*num_target_boxes 
        # C = C.view(bs, num_queries, -1)

        pred_sizes = [len(p) for p in pred_boxes]
        sizes = [len(v) for v in targets]
        indices = [linear_sum_assignment(c.split(pred_sizes, 0)[i]) for i, c in enumerate(C.cpu().split(sizes, -1))]
   
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

if __name__ == '__main__':
    matcher = HungarianMatcher()
    pred_boxes = [torch.rand(3, 4), torch.rand(7, 4)]
    targets = [torch.rand(4, 4), torch.rand(3, 4)]
    print(matcher(pred_boxes, targets)[0])