#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  train_psmnet.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/11 17:38'

import argparse
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
from models.psmnet import psm_net
from util import *
from kitti_util import *


def train_step(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    _train_loss, train_loss = 0, 0
    correct = 0
    total = 0
    train_acc = 0

    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = (inputs[0].to(device), inputs[1].to(device)), targets.to(device)

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

        # count the loss
        _train_loss += loss.item()
        train_loss = _train_loss / (batch_index + 1)

        # calculate acc
        train_acc, correct, total, = calculate_acc(outputs, targets, config, correct, total, is_train = True)

        # log
        if (batch_index + 1) % config.print_interval == 0:
            logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train err: {:6.3f}% | lr: {:.6f}".format(
                batch_index + 1, len(train_loader), train_loss, 100.0 * (1 - train_acc), get_current_lr(optimizer)))

    logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train err: {:6.3f}% | lr: {:.6f}".format(
        batch_index + 1, len(train_loader), train_loss, 100.0 * (1 - train_acc), get_current_lr(optimizer)))

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))

    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_err', 1 - train_acc, epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global writer, best_prec

    net.eval()

    _test_loss, test_loss = 0, 0
    correct = 0
    total = 0
    test_acc = 0

    logger.info(" === Validate ===".format(epoch + 1, config.epochs))

    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = (inputs[0].to(device), inputs[1].to(device)), targets.to(device)
            outputs = net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets)

            # calculate loss and acc
            _test_loss += loss.item()
            test_loss = _test_loss / (batch_index + 1)
            test_acc, correct, total, = calculate_acc(outputs, targets, config, correct, total, is_train = False)

    logger.info("   == test loss: {:.3f} | test err: {:6.3f}%".format(test_loss, 100.0 * (1 - test_acc)))

    writer.add_scalar('test_loss', test_loss, epoch)
    writer.add_scalar('test_err', 1 - test_acc, epoch)
    # Save checkpoint.
    test_acc = 100. * test_loss
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
    """

    Args:
        work_path: `event`, `log`, `checkpoint`保存/读取路径 及 `config.yaml`所在路径

        resume: 是否根据本地ckpt恢复模型，若为Ture，则ckpt文件应该位于`work_path`中
        config_dict:

    Returns:

    """
    global args, writer, logger, config, best_prec

    # 设置路径 work_path
    # set work_path
    args = EasyDict({'work_path': work_path, 'resume': resume})

    # 设置event路径
    # set event path
    writer = SummaryWriter(logdir = args.work_path + '/event')

    # 创建logger用于记录日志
    # create logger and write to log.txt
    logger = Logger(log_file_name = args.work_path + '/log.txt',
                    log_level = logging.DEBUG, logger_name = "KITTI").get_log()

    if config_dict is None:
        # 从yaml文件中读取配置config
        # read config dict from yaml file
        with open(args.work_path + '/config.yaml') as f:
            config_dict = yaml.load(f)

    # 将config转换成EasyDict
    # convert config dict to EasyDict
    config = EasyDict(config_dict)
    logger.info(config)

    # 创建网络模型
    # define netowrk
    net = psm_net(config.max_disparity)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'
    # data parallel for multiple-GPU
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    net.to(device)

    # 设置loss计算函数
    # define loss
    criterion = nn.SmoothL1Loss()

    # 设置optimizer用于反向传播梯度
    # define optimizer
    if config.optimize.type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr = config.lr_scheduler.base_lr,
                              momentum = config.optimize.momentum,
                              weight_decay = config.optimize.weight_decay,
                              nesterov = config.optimize.nesterov)
    elif config.optimize.type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = config.lr_scheduler.base_lr,
                               weight_decay = config.optimize.weight_decay)
    elif config.optimize.type == 'RMSprop':
        optimizer = optim.RMSprop(net.parameters(), lr = config.lr_scheduler.base_lr,
                                  weight_decay = config.optimize.weight_decay)

    # 从checkpoint中恢复网络模型
    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    train_loss = None
    if args.work_path:
        ckpt_file_name = args.work_path + '/' + config.ckpt_name + '.pth.tar'
        if args.resume:
            best_prec, last_epoch = load_checkpoint(ckpt_file_name, net, optimizer = optimizer)

    # 加载训练数据 并进行数据扩增
    # load training data & do data augmentation
    transform_train = transforms.Compose(data_augmentation(config))
    transform_test = transforms.Compose(data_augmentation(config))

    # 得到可用于torch的DataLoader
    # get data loader
    train_loader, test_loader = get_data_loader(transform_train, transform_test, config)

    # 得到用于更新lr的函数
    # get lr scheduler
    lr_scheduler = get_learning_rate_scheduler(optimizer, last_epoch, config)

    # 开始训练
    # start training network
    logger.info("            =======  Training  =======\n")
    for epoch in range(last_epoch + 1, config.epochs):
        # 更新学习率lr
        # adjust learning rate
        if lr_scheduler:
            if config.lr_scheduler.type == 'ADAPTIVE':
                if config.lr_scheduler.mode == 'max':
                    lr_scheduler.step(best_prec, epoch)
                elif config.lr_scheduler.mode == 'min':
                    lr_scheduler.step(train_loss, epoch)
            else:
                lr_scheduler.step(epoch)
        lr = get_current_lr(optimizer)
        writer.add_scalar('learning_rate', lr, epoch)
        # train one epoch
        train_loss, _ = train_step(train_loader, net, criterion, optimizer, epoch, device)
        # validate network
        if epoch == 0 or (epoch + 1) % config.eval_freq == 0 or epoch == config.epochs - 1:
            test(test_loader, net, criterion, optimizer, epoch, device)

    logger.info("======== Training Finished.   best_test_err: {:.3f}% ========".format(100 - best_prec))
    del args, writer, logger, config, last_epoch, best_prec


def parse_args():
    parser = argparse.ArgumentParser(description = 'PyTorch KITTI2015 Dataset Training')
    parser.add_argument('--work-path', required = True, type = str,
                        help = '`event`, `log`, `checkpoint`保存/读取路径 及 `config.yaml`所在路径')
    parser.add_argument('--resume', action = 'store_true',
                        help = 'resume from checkpoint')
    return parser.parse_args()


def main(args):
    start_training(args.work_path, args.resume)


if __name__ == "__main__":
    main(parse_args())
