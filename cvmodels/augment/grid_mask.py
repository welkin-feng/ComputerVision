import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import math


class GridMask(object):
    def __init__(self, num_grid, rotate = 0, ratio = 0.5, mode = 0, prob = 1.):
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h, w = img.shape[-2:]

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        num_grid = np.random.randint(self.num_grid[0], self.num_grid[1] + 1)
        d = min(h, w) // num_grid
        # d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate[0], self.rotate[1] + 1)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = torch.from_numpy(mask).to(img)
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMaskBatch(object):
    def __init__(self, num_grid, rotate = 0, ratio = 0.5, mode = 0, prob = 1.):
        super(GridMaskBatch, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = GridMask(num_grid, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def __call__(self, ori_img_batch, img_batch):
        if np.random.rand() > self.st_prob:
            return img_batch

        n, c, h, w = ori_img_batch.size()
        y = []
        for i in range(n):
            y.append(self.grid(ori_img_batch[i]))
        y = torch.cat(y).view(n, c, h, w).to(ori_img_batch)
        return y
