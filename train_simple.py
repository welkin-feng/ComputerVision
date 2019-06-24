#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  train_simple.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 03:10'

import time
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from easydict import EasyDict
from models import *
from util import *


def train_step(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           config.mixup_alpha, device)

            outputs = net(inputs)
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                # losses for multi classifier
                losses = list(map(criterion, outputs, [targets] * len(outputs)))
                losses = list(map(lambda x, y: x * y, config.classifier_weight, losses))
                loss = sum(losses[:config.num_classifier])
                outputs = outputs[0]
            else:
                loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % config.print_interval == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader),
                train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader),
        train_loss / (batch_index + 1), 100.0 * correct / total, get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))

    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_acc', test_acc, epoch)
    # Save checkpoint.
    test_acc = 100. * correct / total
    state = {
        'state_dict': net.state_dict(),
        'best_prec': best_prec,
        'last_epoch': epoch,
        'optimizer': optimizer.state_dict(),
    }
    is_best = test_acc > best_prec
    save_checkpoint(state, is_best, args.work_path + '/' + config.ckpt_name)
    if is_best:
        best_prec = test_acc


def start_training(work_path, resume = False, config_dict = None):
    global args, writer, logger, config, last_epoch, best_prec
    args = EasyDict({'work_path': work_path, 'resume': resume})
    writer = SummaryWriter(logdir = args.work_path + '/event')
    logger = Logger(log_file_name = args.work_path + '/log.txt',
                    log_level = logging.DEBUG, logger_name = "CIFAR").get_log()

    if config_dict is None:
        # read config from yaml file
        with open(args.work_path + '/config.yaml') as f:
            config_dict = yaml.load(f)
    # convert to dict
    config = EasyDict(config_dict)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    net.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if config.optimize.type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr = config.lr_scheduler.base_lr,
                              momentum = config.optimize.momentum,
                              weight_decay = config.optimize.weight_decay,
                              nesterov = config.optimize.nesterov)
    elif config.optimize.type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = config.lr_scheduler.base_lr,
                               weight_decay = config.optimize.weight_decay)

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer = optimizer)

    # load training data, do data augmentation and get data loader
    transform_train = transforms.Compose(data_augmentation(config))
    transform_test = transforms.Compose(data_augmentation(config, is_train = False))

    train_loader, test_loader = get_data_loader(
        transform_train, transform_test, config)

    # start training
    logger.info("            =======  Training  =======\n")
    lr_scheduler = get_learning_rate_scheduler(optimizer, last_epoch, config)
    for epoch in range(last_epoch + 1, config.epochs):
        # adjust learning rate
        if lr_scheduler:
            if config.lr_scheduler.type == 'ADAPTIVE':
                lr_scheduler.step(best_prec, epoch)
            else:
                lr_scheduler.step(epoch)
        lr = get_current_lr(optimizer)
        writer.add_scalar('learning_rate', lr, epoch)
        train_step(train_loader, net, criterion, optimizer, epoch, device)
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)

    logger.info("======== Training Finished.   best_test_acc: {:.3f}% ========".format(best_prec))


if __name__ == "__main__":
    cfg_dict = {
        'architecture': 'vgg_19',
        'data_path': './data',
        'ckpt_path': './',
        'ckpt_name': 'vgg_19',
        'num_classes': 10,
        'dataset': 'cifar10',
        'use_gpu': True,
        'input_size': 32, 'epochs': 250,
        'batch_size': 128, 'test_batch': 200,
        'eval_freq': 2, 'print_interval': 100,
        'workers': 4,
        'num_classifier': 1,
        'classifier_weight': [1.0],
        'mixup': False, 'mixup_alpha': 0.4,
        'augmentation': {
            'normalize': True, 'random_crop': True,
            'random_horizontal_filp': True,
            'cutout': False, 'holes': 1, 'length': 8
        },
        'optimize': {
            'type': 'SGD', 'weight_decay': 0.0005,
            'momentum': 0.9, 'nesterov': True
        },
        'lr_scheduler': {
            # type: ADAPTIVE or STEP or MultiSTEP or COSINE or HTD
            'type': 'ADAPTIVE', 'base_lr': 0.1,
            'lr_mults': 0.1,  # for ADAPTIVE, STEP and MultiSTEP
            'min_lr': 0.0,  # for ADAPTIVE, COSINE and HTD
            'mode': 'max', 'patience': 10,  # only for ADAPTIVE
            'step_size': 50,  # only for STEP
            'lr_epochs': [50, 100, 150, 200, 250],  # only for MultiSTEP
            'lower_bound': -6.0, 'upper_bound': 3.0
        }
    }
    start_training('./experience/vgg/cifar10', False)
