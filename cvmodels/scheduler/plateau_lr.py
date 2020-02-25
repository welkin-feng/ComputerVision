#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  plateau_lr.py

"""

from torch.optim.lr_scheduler import ReduceLROnPlateau


class PlateauLR(ReduceLROnPlateau):
    def __init__(self, optimizer, mode = 'min', factor = 0.1, patience = 10, cooldown = 0, min_lr = 0,
                 warm_up_steps = 0, warm_up_lr_factor = 0, threshold = 1e-4, threshold_mode = 'rel',
                 eps = 1e-8, verbose = False):
        super(PlateauLR, self).__init__(optimizer, mode, factor, patience, verbose, threshold, threshold_mode,
                                        cooldown, min_lr, eps)
        self.init_lr = [group['lr'] for group in self.optimizer.param_groups]
        self.use_warm_up = False
        self.warm_up_lr = self.init_lr
        self.warm_up_steps = warm_up_steps
        # self.step(True)
        if self.warm_up_steps > 0:
            self.use_warm_up = True
            self.warm_up_lr = [lr * warm_up_lr_factor for lr in self.init_lr]
            for param_group, lr in zip(self.optimizer.param_groups, self.warm_up_lr):
                param_group['lr'] = lr
            self._last_lr = self.warm_up_lr

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def step(self, metrics, epoch = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch < self.warm_up_steps - 1:
            self.use_warm_up = True
            for param_group, lr in zip(self.optimizer.param_groups, self.warm_up_lr):
                param_group['lr'] = lr
        elif self.use_warm_up:
            self.use_warm_up = False
            for param_group, lr in zip(self.optimizer.param_groups, self.init_lr):
                param_group['lr'] = lr
        else:
            if isinstance(metrics, bool):
                is_best = metrics
            else:
                # convert `metrics` to float, in case it's a zero-dim Tensor
                current = float(metrics)
                is_best = self.is_better(current, self.best)
                if is_best:
                    self.best = current
            if is_best:
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode = self.mode, threshold = self.threshold, threshold_mode = self.threshold_mode)
        if hasattr(self, '_last_lr'):
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group['lr'] = lr
