#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/8/29 18:09'

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
from models import *
from util import *
from cifar_util import *


class Trainer():
    """  """

    def __init__(self, work_path, resume = False, config_dict = None):
        """

        Args:
            work_path: `event`, `log`, `checkpoint`保存/读取路径 及 `config.yaml`所在路径
            resume: 是否根据本地ckpt恢复模型，若为Ture，则ckpt文件应该位于`work_path`中
            config_dict:

        """
        # 设置路径 work_path
        # set work_path
        self.args = EasyDict({'work_path': work_path, 'resume': resume})
        # 创建logger用于记录日志
        # create logger and write to log.txt
        self.logger = Logger(log_file_name = self.args.work_path + '/log.txt',
                             log_level = logging.DEBUG, logger_name = "CIFAR").get_log()
        # 设置event路径
        # set event path
        self.writer = SummaryWriter(logdir = self.args.work_path + '/event')

        if config_dict is None:
            # 从yaml文件中读取配置config
            # read config dict from yaml file
            with open(self.args.work_path + '/config.yaml') as f:
                config_dict = yaml.load(f)
        # 将config转换成EasyDict
        # convert config dict to EasyDict
        self.config = EasyDict(config_dict)

        # 创建网络模型
        # define netowrk
        self.net = get_model(self.config)

        # CPU or GPU
        self.device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
        # data parallel for multiple-GPU
        if self.device == 'cuda':
            self.net = nn.DataParallel(self.net)
            cudnn.benchmark = True
        self.net.to(self.device)

        # 设置loss计算函数
        # define loss
        self.criterion = nn.CrossEntropyLoss()

        # 设置optimizer用于反向传播梯度
        # define optimizer
        if self.config.optimize.type == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr = self.config.lr_scheduler.base_lr,
                                       momentum = self.config.optimize.momentum,
                                       weight_decay = self.config.optimize.weight_decay,
                                       nesterov = self.config.optimize.nesterov)
        elif self.config.optimize.type == 'Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr = self.config.lr_scheduler.base_lr,
                                        weight_decay = self.config.optimize.weight_decay)

        # 从checkpoint中恢复网络模型
        # resume from a checkpoint
        self.last_epoch = -1
        self.best_prec = 0
        if self.args.work_path:
            ckpt_file_name = self.args.work_path + '/' + self.config.ckpt_name + '.pth.tar'
            if self.args.resume:
                self.best_prec, self.last_epoch = load_checkpoint(
                    ckpt_file_name, self.net, optimizer = self.optimizer)

        # 得到用于更新lr的函数
        # get lr scheduler
        self.lr_scheduler = get_learning_rate_scheduler(self.optimizer, self.last_epoch, self.config)

        self.logger.info(self.config)
        self.logger.info(self.net)
        self.logger.info(" == total parameters: " + str(count_parameters(self.net)))

    def start_training(self):

        # 加载训练数据 并进行数据扩增
        # load training data & do data augmentation
        transform_train = transforms.Compose(data_augmentation(self.config))
        transform_test = transforms.Compose(data_augmentation(self.config, is_train = False))

        # 得到可用于torch的DataLoader
        # get data loader
        train_loader, test_loader = get_data_loader(transform_train, transform_test, self.config)

        # 开始训练
        # start training network
        self.logger.info("            =======  Training  =======\n")

        train_loss = None
        for epoch in range(self.last_epoch + 1, self.config.epochs):
            # 更新学习率lr
            # adjust learning rate
            if self.lr_scheduler:
                if self.config.lr_scheduler.type == 'ADAPTIVE':
                    if self.config.lr_scheduler.mode == 'max':
                        self.lr_scheduler.step(self.best_prec, epoch)
                    elif self.config.lr_scheduler.mode == 'min' and train_loss is not None:
                        self.lr_scheduler.step(train_loss, epoch)
                else:
                    self.lr_scheduler.step(epoch)
            lr = get_current_lr(self.optimizer)
            self.writer.add_scalar('learning_rate', lr, epoch)
            # train one epoch
            train_loss, _ = self.train_step(train_loader, epoch)
            # validate network
            if epoch == 0 or (epoch + 1) % self.config.eval_freq == 0 or epoch == self.config.epochs - 1:
                self.test(test_loader, epoch)

        self.logger.info("======== Training Finished.   best_test_acc: {:.3f}% ========".format(self.best_prec))

    def train_step(self, train_loader, epoch):

        start = time.time()
        self.net.train()

        _train_loss, train_loss = 0, 0
        correct, total, train_acc = 0, 0, 0

        self.logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, self.config.epochs))

        for batch_index, (inputs, targets) in enumerate(train_loader):
            # move tensor to GPU
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.config.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                               self.config.mixup_alpha, self.device)
                outputs = self.net(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.net(inputs)
                if isinstance(outputs, tuple):
                    # losses for multi classifier
                    losses = list(map(self.criterion, outputs, [targets] * len(outputs)))
                    losses = list(map(lambda x, y: x * y, self.config.classifier_weight, losses))
                    loss = sum(losses[:self.config.num_classifier])
                    outputs = outputs[0]
                else:
                    loss = self.criterion(outputs, targets)

            # zero the gradient buffers
            self.optimizer.zero_grad()
            # backward
            loss.backward()
            # update weight
            self.optimizer.step()

            # count the loss
            _train_loss += loss.item()
            train_loss = _train_loss / (batch_index + 1)

            # calculate acc
            if self.config.mixup:
                train_acc, correct, total, = calculate_acc(outputs, targets, self.config, correct, total,
                                                           is_train = True, lam = lam,
                                                           targets_a = targets_a, targets_b = targets_b)
            else:
                train_acc, correct, total, = calculate_acc(outputs, targets, self.config, correct, total,
                                                           is_train = True)
            # log
            if (batch_index + 1) % self.config.print_interval == 0:
                self.logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.2e}".format(
                    batch_index + 1, len(train_loader), train_loss, 100.0 * train_acc, get_current_lr(self.optimizer)))

        end = time.time()
        self.logger.info("   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.2e}".format(
            batch_index + 1, len(train_loader), train_loss, 100.0 * train_acc, get_current_lr(self.optimizer)))
        self.logger.info("   == cost time: {:.4f}s".format(end - start))
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('train_acc', train_acc, epoch)

        return train_loss, train_acc

    def test(self, test_loader, epoch):

        self.net.eval()

        _test_loss, test_loss = 0, 0
        correct, total, test_acc = 0, 0, 0

        self.logger.info(" === Validate ===".format(epoch + 1, self.config.epochs))

        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, targets)

                # calculate loss and acc
                _test_loss += loss.item()
                test_loss = _test_loss / (batch_index + 1)
                test_acc, correct, total, = calculate_acc(outputs, targets, self.config, correct, total,
                                                          is_train = False)

        self.logger.info("   == test loss: {:.3f} | test acc: {:6.3f}%".format(test_loss, 100.0 * test_acc))
        self.writer.add_scalar('test_loss', test_loss, epoch)
        self.writer.add_scalar('test_acc', test_acc, epoch)

        # Save checkpoint.
        test_acc = 100. * test_acc
        is_best = test_acc > self.best_prec
        if is_best:
            self.best_prec = test_acc
        state = {
            'state_dict': self.net.state_dict(),
            'best_prec': self.best_prec,
            'last_epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        save_checkpoint(state, is_best, self.args.work_path + '/' + self.config.ckpt_name)


def parse_args():
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR Dataset Training')
    parser.add_argument('--work-path', required = True, type = str)
    parser.add_argument('--resume', action = 'store_true',
                        help = 'resume from checkpoint')
    return parser.parse_args()


def main(args):
    trainer = Trainer(args.work_path, args.resume)
    trainer.start_training()


if __name__ == "__main__":
    main(parse_args())