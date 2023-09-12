from ..registry import DETECTORS
from .single_stage_mem import SingleStageMem


@DETECTORS.register_module
class SipMaskMem(SingleStageMem):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 checkpoint=False):
        super(SipMaskMem, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, checkpoint)