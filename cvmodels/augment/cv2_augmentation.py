import cv2
import random
import numpy as np
import torch


# helper --
# def make_grid_image(width, height, grid_size = 16):
#     image = np.zeros((height, width), np.float32)
#     for y in range(0, height, 2 * grid_size):
#         for x in range(0, width, 2 * grid_size):
#             image[y: y + grid_size, x:x + grid_size] = 1
#
#     # for y in range(height+grid_size,2*grid_size):
#     #     for x in range(width+grid_size,2*grid_size):
#     #          image[y: y+grid_size,x:x+grid_size] = 1
#
#     return image


# ---
def do_identity(image, magnitude=None):
    return image


# *** geometric ***

def do_random_projective(image, magnitude=0.2):
    mag = np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    x0, y0 = 0, 0
    x1, y1 = 1, 0
    x2, y2 = 1, 1
    x3, y3 = 0, 1

    mode = np.random.choice(['top', 'bottom', 'left', 'right'])
    if mode == 'top':
        x0, x1 = x0 + mag, x1 - mag
    if mode == 'bottom':
        x3, x2 = x3 + mag, x2 - mag
    if mode == 'left':
        y0, y3 = y0 + mag, y3 - mag
    if mode == 'right':
        y1, y2 = y1 + mag, y2 - mag

    s = np.array([[0, 0], [1, 0], [1, 1], [0, 1], ]) * [[width, height]]
    d = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3], ]) * [[width, height]]
    transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

    image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_perspective(image, magnitude=0.1):
    mag = np.random.uniform(-1, 1, (4, 2)) * magnitude

    height, width = image.shape[:2]
    s = np.array([[0, 0], [1, 0], [1, 1], [0, 1], ])
    d = s + mag
    s *= [[width, height]]
    d *= [[width, height]]
    transform = cv2.getPerspectiveTransform(s.astype(np.float32), d.astype(np.float32))

    image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


def do_random_scale(image, magnitude=0.2):
    s = 1 + np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [s, 0, 0],
        [0, s, 0],
    ], np.float32)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_shear_x(image, magnitude=0.2):
    sx = np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1, sx, 0],
        [0, 1, 0],
    ], np.float32)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_shear_y(image, magnitude=0.1):
    sy = np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1, 0, 0],
        [sy, 1, 0],
    ], np.float32)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_x(image, magnitude=0.2):
    sx = 1 + np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [sx, 0, 0],
        [0, 1, 0],
    ], np.float32)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_stretch_y(image, magnitude=0.2):
    sy = 1 + np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    transform = np.array([
        [1, 0, 0],
        [0, sy, 0],
    ], np.float32)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_random_rotate(image, magnitude=15):
    angle = np.random.uniform(-1, 1) * magnitude

    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2

    transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


# ----
def do_random_grid_distortion(image, magnitude=0.3):
    num_step = 5
    distort = magnitude

    # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    distort_x = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]
    distort_y = [1 + random.uniform(-distort, distort) for i in range(num_step + 1)]

    # ---
    height, width = image.shape[:2]
    xx = np.zeros(width, np.float32)
    step_x = width // num_step

    prev = 0
    for i, x in enumerate(range(0, width, step_x)):
        start = x
        end = x + step_x
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + step_x * distort_x[i]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    yy = np.zeros(height, np.float32)
    step_y = height // num_step
    prev = 0
    for idx, y in enumerate(range(0, height, step_y)):
        start = y
        end = y + step_y
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + step_y * distort_y[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image


# https://github.com/albumentations-team/albumentations/blob/8b58a3dbd2f35558b3790a1dbff6b42b98e89ea5/albumentations/augmentations/transforms.py

# https://ciechanow.ski/mesh-transforms/
# https://stackoverflow.com/questions/53907633/how-to-warp-an-image-using-deformed-mesh
# http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
def do_random_custom_distortion1(image, magnitude=0.15):
    distort = magnitude

    height, width = image.shape
    s_x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    s_y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
    d_x = s_x.copy()
    d_y = s_y.copy()
    d_x[[1, 4, 7]] += np.random.uniform(-distort, distort, 3)
    d_y[[3, 4, 5]] += np.random.uniform(-distort, distort, 3)

    s_x = (s_x * width)
    s_y = (s_y * height)
    d_x = (d_x * width)
    d_y = (d_y * height)

    # ---
    distort = np.zeros((height, width), np.float32)
    for index in ([4, 1, 3], [4, 1, 5], [4, 7, 3], [4, 7, 5]):
        point = np.stack([s_x[index], s_y[index]]).T
        qoint = np.stack([d_x[index], d_y[index]]).T

        src = np.array(point, np.float32)
        dst = np.array(qoint, np.float32)
        mat = cv2.getAffineTransform(src, dst)

        point = np.round(point).astype(np.int32)
        x0 = np.min(point[:, 0])
        x1 = np.max(point[:, 0])
        y0 = np.min(point[:, 1])
        y1 = np.max(point[:, 1])
        mask = np.zeros((height, width), np.float32)
        mask[y0:y1, x0:x1] = 1

        mask = mask * image
        warp = cv2.warpAffine(mask, mat, (width, height), borderMode=cv2.BORDER_REPLICATE)
        distort = np.maximum(distort, warp)
        # distort = distort+warp

    return distort


# *** intensity ***
def do_random_contast(image, magnitude=0.2):
    alpha = 1 + random.uniform(-1, 1) * magnitude
    image = image.astype(np.float32) * alpha
    image = np.clip(image, 0, 1)
    return image


def do_random_block_fade(image, magnitude=0.3):
    size = [0.1, magnitude]

    height, width = image.shape

    # get bounding box
    m = image.copy()
    cv2.rectangle(m, (0, 0), (height, width), 1, 5)
    m = image < 0.5
    if m.sum() == 0:
        return image

    m = np.where(m)
    y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
    w = x1 - x0
    h = y1 - y0
    if w * h < 10:
        return image

    ew, eh = np.random.uniform(*size, 2)
    ew = int(ew * w)
    eh = int(eh * h)

    ex = np.random.randint(0, w - ew) + x0
    ey = np.random.randint(0, h - eh) + y0

    image[ey:ey + eh, ex:ex + ew] *= np.random.uniform(0.1, 0.5)  # 1 #
    image = np.clip(image, 0, 1)
    return image


# *** noise ***
# https://www.kaggle.com/ren4yu/bengali-morphological-ops-as-image-augmentation
def do_random_erode(image, magnitude=2):
    s = int(round(1 + np.random.uniform(0, 1) * magnitude))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
    image = cv2.erode(image, kernel, iterations=1)
    return image


def do_random_dilate(image, magnitude=1.5):
    s = int(round(1 + np.random.uniform(0, 1) * magnitude))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
    image = cv2.dilate(image, kernel, iterations=1)
    return image


def do_random_sprinkle(image, magnitude=0.2):
    size = 16
    num_sprinkle = int(round(1 + np.random.randint(10) * magnitude))

    height, width = image.shape
    image = image.copy()
    image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
    m = np.where(image_small > 0.25)
    num = len(m[0])
    if num == 0:
        return image

    s = size // 2
    i = np.random.choice(num, num_sprinkle)
    for y, x in zip(m[0][i], m[1][i]):
        y = y * 4 + 2
        x = x * 4 + 2
        image[y - s:y + s, x - s:x + s] = 0  # 0.5 #1 #
    return image


# https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
def do_random_noise(image, magnitude=0.15):
    height, width = image.shape
    noise = np.random.uniform(-1, 1, (height, width)) * magnitude
    image = image + noise
    image = np.clip(image, 0, 1)
    return image


def do_random_line(image, magnitude=0.2):
    num_lines = int(round(1 + np.random.randint(10) * magnitude))

    height, width = image.shape
    image = image.copy()

    def line0():
        return (0, 0), (width - 1, 0)

    def line1():
        return (0, height - 1), (width - 1, height - 1)

    def line2():
        return (0, 0), (0, height - 1)

    def line3():
        return (width - 1, 0), (width - 1, height - 1)

    def line4():
        x0, x1 = np.random.choice(width, 2)
        return (x0, 0), (x1, height - 1)

    def line5():
        y0, y1 = np.random.choice(height, 2)
        return (0, y0), (width - 1, y1)

    for i in range(num_lines):
        p = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 1, 1])
        func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p / p.sum())
        (x0, y0), (x1, y1) = func()

        color = np.random.uniform(0, 1)
        thickness = np.random.randint(1, 5)
        line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

        cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

    return image


# batch augmentation that uses pairing, e.g mixup, cutmix, cutout #####################
def make_object_box(image):
    m = image.copy()
    cv2.rectangle(m, (0, 0), (236, 137), 0, 10)
    m = m - np.min(m)
    m = m / np.max(m)
    h = m < 0.5

    row = np.any(h, axis=1)
    col = np.any(h, axis=0)
    y0, y1 = np.where(row)[0][[0, -1]]
    x0, x1 = np.where(col)[0][[0, -1]]

    return [x0, y0], [x1, y1]


def do_random_batch_mixup(input, onehot):
    batch_size = len(input)

    alpha = 0.4  # 0.2  #0.2,0.4
    gamma = np.random.beta(alpha, alpha, batch_size)
    gamma = np.maximum(1 - gamma, gamma)

    # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
    gamma = torch.from_numpy(gamma).float().to(input.device)
    perm = torch.randperm(batch_size).to(input.device)
    perm_input = input[perm]
    perm_onehot = [t[perm] for t in onehot]

    gamma = gamma.view(batch_size, 1, 1, 1)
    mix_input = gamma * input + (1 - gamma) * perm_input
    gamma = gamma.view(batch_size, 1)
    mix_onehot = [gamma * t + (1 - gamma) * perm_t for t, perm_t in zip(onehot, perm_onehot)]

    return mix_input, mix_onehot, (perm_input, perm_onehot)


def do_random_batch_cutout(input, onehot):
    batch_size, C, H, W = input.shape

    mask = np.ones((batch_size, C, H, W), np.float32)
    for b in range(batch_size):
        length = int(np.random.uniform(0.1, 0.5) * min(H, W))
        y = np.random.randint(H)
        x = np.random.randint(W)

        y0 = np.clip(y - length // 2, 0, H)
        y1 = np.clip(y + length // 2, 0, H)
        x0 = np.clip(x - length // 2, 0, W)
        x1 = np.clip(x + length // 2, 0, W)
        mask[b, :, y0: y1, x0: x1] = 0
    mask = torch.from_numpy(mask).to(input.device)

    input = input * mask
    return input, onehot, None
