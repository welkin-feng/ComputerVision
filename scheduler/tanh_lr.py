#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  tanh_lr.py

"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class TanhLR(_LRScheduler):
    """
       Hyberbolic-Tangent decay with restarts.
       This is described in the paper https://arxiv.org/abs/1806.01593
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lower_bound: float = -6.,
                 upper_bound: float = 4.,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t = 0,
                 warmup_lr_init = 0,
                 warmup_prefix = False,
                 cycle_limit = 0,
                 t_in_epochs = True,
                 last_epoch = -1) -> None:

        assert t_initial > 0
        assert lr_min >= 0
        assert lower_bound < upper_bound
        assert cycle_limit >= 0
        assert warmup_t >= 0
        assert warmup_lr_init >= 0
        self.lb = lower_bound
        self.ub = upper_bound
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs

        super(TanhLR, self).__init__(optimizer, last_epoch)

        if self.warmup_t:
            t_v = self.base_lrs if self.warmup_prefix else self.get_lr(self.warmup_t)
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in t_v]
            for param_group, value in zip(self.optimizer.param_groups, self.warmup_lr_init):
                param_group['lr'] = value
        else:
            self.warmup_steps = [1 for _ in self.base_lrs]

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                gamma = self.decay_rate ** i
                lr_min = self.lr_min * gamma
                lr_max_values = [v * gamma for v in self.base_lrs]

                tr = t_curr / t_i
                lrs = [lr_min + 0.5 * (lr_max - lr_min) * (1 - math.tanh(self.lb * (1. - tr) + self.ub * tr))
                    for lr_max in lr_max_values]
            else:
                lrs = [self.lr_min * (self.decay_rate ** self.cycle_limit) for _ in self.base_lrs]
        return lrs
