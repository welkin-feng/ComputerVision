#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  frn.py

"""
import torch
from torch import nn


class FRN(nn.Module):
    def __init__(self, num_features, eps = 1e-6, is_eps_leanable = False):
        super(FRN, self).__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad = True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad = True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad = True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim = [2, 3], keepdim = True)
        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        # Scale and Bias
        x = self.weight * x + self.bias
        return x


class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad = True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)
