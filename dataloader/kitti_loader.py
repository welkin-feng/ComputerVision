#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  kitti_loader.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/12 18:26'

import os
import numbers
import torch
import numpy as np
import torchvision.transforms.functional as F

from os.path import join
from torch.utils.data import Dataset
from PIL import Image

__all__ = ['KITTI2015', 'RandomCrop', 'ToTensor', 'Normalize']


class KITTI2015(Dataset):

    def __init__(self, root, mode = 'train', validate_size = 40, occ = True,
                 transform = None, target_transform = None, download = False):
        super().__init__()

        self.root = os.path.expanduser(root)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train' or mode == 'validate':
            self.base_folder = 'training'
        elif mode == 'test':
            self.base_folder = 'testing'

        self.left_dir = join(self.root, self.base_folder, 'image_2')
        self.right_dir = join(self.root, self.base_folder, 'image_3')
        left_imgs_list = list()
        right_imgs_list = list()

        if mode == 'train':
            imgs_range = range(200 - validate_size)
        elif mode == 'validate':
            imgs_range = range(200 - validate_size, 200)
        elif mode == 'test':
            imgs_range = range(200)

        fmt = '{:06}_10.png'

        for i in imgs_range:
            left_imgs_list.append(join(self.left_dir, fmt.format(i)))
            right_imgs_list.append(join(self.right_dir, fmt.format(i)))

        self.left_imgs_list = left_imgs_list
        self.right_imgs_list = right_imgs_list

        if mode == 'train' or mode == 'validate':
            disp_imgs_list = list()
            if occ:
                disp_dir = join(self.root, self.base_folder, 'disp_occ_0')
            else:
                disp_dir = join(self.root, self.base_folder, 'disp_noc_0')
            disp_fmt = '{:06}_10.png'
            for i in imgs_range:
                disp_imgs_list.append(join(disp_dir, disp_fmt.format(i)))

            self.disp_imgs_list = disp_imgs_list

    def __len__(self):
        return len(self.left_imgs_list)

    def __getitem__(self, index):
        # return a PIL Image
        data = {'left': Image.open(self.left_imgs_list[index]),
                'right': Image.open(self.right_imgs_list[index])}

        if self.mode != 'test':
            data['disp'] = Image.open(self.disp_imgs_list[index])

        if self.transform:
            data = self.transform(data)

        left_img, right_img, disp = data['left'], data['right'], data['disp']

        return (left_img, right_img), disp


class RandomCrop():

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        new_h, new_w = self.size
        if isinstance(sample['left'], Image.Image):
            w, h = sample['left'].size
        else:
            h, w, _ = sample['left'].shape
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for key in sample:
            if isinstance(sample[key], Image.Image):
                sample[key] = sample[key].crop((left, top, left + new_w, top + new_h))
            else:
                sample[key] = sample[key][top: top + new_h, left: left + new_w]

        return sample


class ToTensor():

    def __call__(self, sample):
        # PIL.Image.Image W x H x C ---> torch.Tensor C x H x W
        sample['left'] = F.to_tensor(sample['left']).type(torch.FloatTensor)
        sample['right'] = F.to_tensor(sample['right']).type(torch.FloatTensor)

        if 'disp' in sample:
            sample['disp'] = F.to_tensor(sample['disp'])
            if sample['disp'].max() > 255:
                sample['disp'] = sample['disp'] / 255
            sample['disp'] = sample['disp'].type(torch.FloatTensor)

        return sample


class Normalize():
    '''
    RGB mode
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = F.normalize(sample['left'], self.mean, self.std)
        sample['right'] = F.normalize(sample['right'], self.mean, self.std)

        return sample


if __name__ == '__main__':
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # BGR
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    train_transform = T.Compose([RandomCrop([256, 512]), ToTensor()])
    train_dataset = KITTI2015('D:/dataset/data_scene_flow', mode = 'train', transform = train_transform)
    train_loader = DataLoader(train_dataset)
    print(len(train_loader))

    # test_transform = T.Compose([ToTensor()])
    # test_dataset = KITTI2015('D:/dataset/data_scene_flow', mode='test', transform=test_transform)

    # validate_transform = T.Compose([ToTensor()])
    # validate_dataset = KITTI2015('D:/dataset/data_scene_flow', mode='validate', transform=validate_transform)

    # datasets = [train_dataset, test_dataset, validate_dataset]

    # for i, dataset in enumerate(datasets):
    #     a = dataset[0]['right'].numpy().transpose([1, 2, 0])
    #     plt.subplot(3, 1, i + 1)
    #     plt.imshow(a)
    # plt.show()
