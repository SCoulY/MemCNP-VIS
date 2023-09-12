import torch

from .hook import Hook


class MeMCNPHook(Hook):
    def __init__(self, before_epoch=True, after_epoch=True, after_iter=True):
        self._before_epoch = before_epoch
        self._after_epoch = after_epoch
        self._after_iter = after_iter

    def before_epoch(self, runner):
        if not runner._epoch and self._before_epoch:
            pass
            
    def after_iter(self, runner):
        if self._after_iter and runner.iter%50==0:
            runner.model.module.bbox_head.mem.offline_update()
            

    def after_epoch(self, runner):
        if self._after_epoch:
            runner.model.module.bbox_head.mem.offline_update()
