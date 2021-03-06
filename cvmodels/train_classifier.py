#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision

File Name:  train_classifier.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:05'

import argparse
from cvmodels.trainer import ClassificationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Dataset Training')
    parser.add_argument('--work-path', required=True, type=str)
    parser.add_argument('--resume', action='store_true',
                        help='resume from checkpoint')
    return parser.parse_args()


def main(args):
    trainer = ClassificationTrainer(args.work_path, args.resume)
    trainer.start_training()


if __name__ == "__main__":
    main(parse_args())
