#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  train_alter.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 01:51'

import torch
import torch.nn as nn


dtype = torch.float32


def check_accuracy_part5(loader, model, device, verbose = False):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device = device, dtype = dtype)  # move to device, e.g. GPU
            y = y.to(device = device, dtype = torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if verbose:
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def train_part5(loader_train, model, optimizer, device, it = 400, print_every = 100, verbose = False):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device = device)  # move the model parameters to CPU/GPU

    loss_list = []
    acc_list = []
    for t, (x, y) in enumerate(loader_train):
        model.train()  # put model to training mode
        x = x.to(device = device, dtype = dtype)  # move to device, e.g. GPU
        y = y.to(device = device, dtype = torch.long)

        scores = model(x)
        loss_ = nn.CrossEntropyLoss()
        loss = loss_(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()
        if t % print_every == 0:
            if verbose:
                _, preds = scores.max(1)
                num_correct = (preds == y).sum()
                num_samples = preds.size(0)
                train_acc = float(num_correct) / num_samples
                print('Iteration %d, loss = %.4f, train_acc = %.2f' % (t, loss.item(), 100 * train_acc))
            loss_list.append(loss.item())
            acc = check_accuracy_part5(loader_val, model, verbose)
            acc_list.append(acc)

        if t == it:
            return loss_list, acc_list

    if verbose:
        _, preds = scores.max(1)
        num_correct = (preds == y).sum()
        num_samples = preds.size(0)
        train_acc = float(num_correct) / num_samples
        print('Iteration %d, loss = %.4f, train_acc = %.2f' % (t, loss.item(), 100 * train_acc))
        check_accuracy_part5(loader_val, model, verbose)

    return loss_list, acc_list
