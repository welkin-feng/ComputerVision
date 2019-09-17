#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  util.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 01:49'

import os
import math
import shutil
import logging

import torch
import torch.optim as optim

__all__ = ['Logger', 'count_parameters',
           'save_checkpoint', 'load_checkpoint',
           'adjust_learning_rate', 'get_learning_rate_scheduler']


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # formatter = logging.Formatter('[%(asctime)s] - [%(filename)s line:%(lineno)d] : %(message)s')
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s] : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(path, model, optimizer = None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict = False)

        if optimizer is not None:
            best_prec = checkpoint['best_prec']
            last_epoch = checkpoint['last_epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("=== done. also loaded optimizer from checkpoint '{}' (epoch {}) ===".format(
                path, last_epoch + 1))
            return best_prec, last_epoch


# todo create class `HTD`, and remove this function
def adjust_learning_rate(optimizer, epoch, config):
    lr = optimizer.param_groups[0]['lr']
    if config.lr_scheduler.type == 'STEP':
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == 'COSINE':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
             (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
             (1.0 + math.cos(math.pi * ratio)) / 2.0
    elif config.lr_scheduler.type == 'HTD':
        ratio = epoch / config.epochs
        lr = config.lr_scheduler.min_lr + \
             (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr) * \
             (1.0 - math.tanh(config.lr_scheduler.lower_bound +
                              (config.lr_scheduler.upper_bound - config.lr_scheduler.lower_bound) * ratio)) / 2.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_learning_rate_scheduler(optimizer, last_epoch, config):
    lr_scheduler = None
    if config.lr_scheduler.type == 'ADAPTIVE':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode = config.lr_scheduler.mode,
                                                            factor = config.lr_scheduler.lr_mults,
                                                            patience = config.lr_scheduler.patience,
                                                            threshold_mode = 'rel', threshold = 0.0001,
                                                            min_lr = 0, eps = 1e-8)
    elif config.lr_scheduler.type == 'STEP':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                 step_size = config.lr_scheduler.step_size,
                                                 gamma = config.lr_scheduler.lr_mults,
                                                 last_epoch = last_epoch)
    elif config.lr_scheduler.type == 'MultiSTEP':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones = config.lr_scheduler.lr_epochs,
                                                      gamma = config.lr_scheduler.lr_mults,
                                                      last_epoch = last_epoch)
    elif config.lr_scheduler.type == 'COSINE':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max = config.epochs,
                                                            eta_min = config.lr_scheduler.min_lr,
                                                            last_epoch = last_epoch)

    return lr_scheduler
