# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import random
import cv2
import math
from PIL import Image, ImageEnhance, ImageFilter


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           lambda: (tf.constant(target_shortside_len), target_shortside_len * w//h),
                           lambda: (target_shortside_len * h//w,  tf.constant(target_shortside_len)))
    '''
    if tf.less(h, w) is not None:
        new_h = target_shortside_len
        new_w = target_shortside_len * w // h
    else:
        new_h = target_shortside_len *h // w
        new_w = target_shortside_len
    '''

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)

    x1, x2, x3, x4 = x1 * new_w//w, x2 * new_w//w, x3 * new_w//w, x4 * new_w//w
    y1, y2, y3, y4 = y1 * new_h//h, y2 * new_h//h, y3 * new_h//h, y4 * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               lambda: (tf.constant(target_shortside_len), target_shortside_len * w // h),
                               lambda: (target_shortside_len * h // w, tf.constant(target_shortside_len)))
        '''
        new_h, new_w = tf.cond(tf.less(h, w),
                               lambda: (target_shortside_len, target_shortside_len*w//h),
                               lambda: (target_shortside_len*h//w, target_shortside_len))
        '''
        '''
        if tf.less(h, w) is not None:
            new_h = target_shortside_len
            new_w = target_shortside_len * w // h
        else:
            new_h = target_shortside_len * h // w
            new_w = target_shortside_len
        '''

        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]


def random_flip_left_right(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    coin = np.random.rand()
    if coin > 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)

        x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)
        new_x1 = w - x1
        new_x2 = w - x2
        new_x3 = w - x3
        new_x4 = w - x4
        return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label], axis=0))
    else:
        return img_tensor,  gtboxes_and_label


def color(img):
    image = Image.fromarray(img.astype(np.uint8))

    if random.choice([0, 1]):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_bri.enhance(brightness)

    if random.choice([0, 1]):
        enh_col = ImageEnhance.Color(image)
        color = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_col.enhance(color)

    if random.choice([0, 1]):
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_con.enhance(contrast)

    if random.choice([0, 1]):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_sha.enhance(sharpness)

    if random.choice([0, 1]):
        image = image.filter(ImageFilter.BLUR)

    return np.array(image, dtype=np.float32)


def color_aug(img_tensor):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.py_func(color, inp=[img_tensor], Tout=tf.float32)
    img_tensor = tf.reshape(img_tensor, [h, w, 3])
    return img_tensor


def crop(img, gtboxes_and_label):

    if random.choice([0, 1]):
        h, w = img.shape[0], img.shape[1]

        min_loc_x = np.min(gtboxes_and_label[:, [0, 2, 4, 6]])
        max_loc_x = np.max(gtboxes_and_label[:, [0, 2, 4, 6]])
        min_loc_y = np.min(gtboxes_and_label[:, [1, 3, 5, 7]])
        max_loc_y = np.max(gtboxes_and_label[:, [1, 3, 5, 7]])

        crop_x1 = int(np.random.uniform(0, max(0, min_loc_x-10)))
        crop_y1 = int(np.random.uniform(0, max(0, min_loc_y-10)))
        crop_x2 = int(np.random.uniform(min(max_loc_x+10, w), w))
        crop_y2 = int(np.random.uniform(min(max_loc_y+10, h), h))

        img = cv2.resize(img[crop_y1:crop_y2, crop_x1:crop_x2, :], (w, h))
        gtboxes_and_label[:, [0, 2, 4, 6]] -= crop_x1
        gtboxes_and_label[:, [1, 3, 5, 7]] -= crop_y1
        gtboxes_and_label[:, [0, 2, 4, 6]] = np.int32(gtboxes_and_label[:, [0, 2, 4, 6]] * w / (crop_x2 - crop_x1))
        gtboxes_and_label[:, [1, 3, 5, 7]] = np.int32(gtboxes_and_label[:, [1, 3, 5, 7]] * h / (crop_y2 - crop_y1))

    return img, gtboxes_and_label


def random_crop(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor, gtboxes_and_label = tf.py_func(crop, inp=[img_tensor, gtboxes_and_label], Tout=[tf.float32, tf.int32])
    img_tensor = tf.reshape(img_tensor, [h, w, 3])
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
    return img_tensor, gtboxes_and_label


def rotate(img, gtboxes_and_label):
    if random.choice([0, 1]):
        h, w = img.shape[0], img.shape[1]
        angle = random.choice([-30, -20, -10, 0, 10, 20, 30])
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_img = cv2.warpAffine(img, M, (w, h))
        rotated_gtboxes_and_label = np.zeros(shape=gtboxes_and_label.shape, dtype=np.int32)
        out_row = set()
        for i in range(gtboxes_and_label.shape[0]):
            for j in range(0, gtboxes_and_label.shape[1], 2):
                if j==8:
                    rotated_gtboxes_and_label[i][j] = gtboxes_and_label[i][j]
                else:
                    loc1, loc2 = M * np.mat([[gtboxes_and_label[i][j]], [gtboxes_and_label[i][j+1]], [1]])
                    rotated_gtboxes_and_label[i][j] = math.ceil(loc1)
                    rotated_gtboxes_and_label[i][j+1] = math.ceil(loc2)
                    if loc1<0 or loc1>w or loc2<0 or loc2>h:
                        out_row.add(i)
        img = rotated_img
        gtboxes_and_label = rotated_gtboxes_and_label
        if gtboxes_and_label.shape[0] != len(out_row):
            gtboxes_and_label = np.delete(rotated_gtboxes_and_label, list(out_row), 0)
    return img, gtboxes_and_label

def random_rotate(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor, gtboxes_and_label = tf.py_func(rotate, inp=[img_tensor, gtboxes_and_label], Tout=[tf.float32, tf.int32])
    img_tensor = tf.reshape(img_tensor, [h, w, 3])
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
    return img_tensor, gtboxes_and_label


def size(img, gtboxes_and_label, min_ratio, max_ratio):
    
    h, w = img.shape[0], img.shape[1]
    ratio = np.random.uniform(min_ratio, max_ratio)
    new_h, new_w = int(ratio * img.shape[0]), int(ratio * img.shape[1])
    resized_img = cv2.resize(img, (new_w, new_h))
    size_gt = np.copy(gtboxes_and_label)
    out_row = []

    if ratio <= 1.0:
        img_x = resized_img.shape[0]
        img_y = resized_img.shape[1]
        img2 = np.zeros((h, w, 3), dtype=np.float32)
        img_x_padding = (h - img_x) // 2
        img_y_padding = (w - img_y) // 2
        img2[img_x_padding:img_x_padding + img_x, img_y_padding:img_y_padding + img_y, :] = resized_img[:, :, :]
        size_img = img2

        size_gt[:, :8] = gtboxes_and_label[:, :8] * ratio
        size_gt[:, 1:8:2] += img_x_padding
        size_gt[:, 0:8:2] += img_y_padding

    else:
        img_x = resized_img.shape[0]
        img_y = resized_img.shape[1]
        img2 = np.zeros((h, w, 3), dtype=np.float32)
        img_x_padding = (img_x - h) // 2
        img_y_padding = (img_y - w) // 2
        img2[:, :, :] = resized_img[img_x_padding:img_x_padding + h, img_y_padding:img_y_padding + w, :]
        size_img = img2

        size_gt[:, :8] = gtboxes_and_label[:, :8] * ratio
        size_gt[:, 1:8:2] -= img_x_padding
        size_gt[:, 0:8:2] -= img_y_padding
        for i in range(size_gt.shape[0]):
            for j in range(0, size_gt.shape[1]-1, 2):
                if size_gt[i, j]<0 or size_gt[i, j]>w or size_gt[i, j+1] < 0 or size_gt[i, j+1] > h:
                    out_row.append(i)
                    break
        if size_gt.shape[0] != len(out_row):
            size_gt = np.delete(size_gt, out_row, 0)

    return size_img, size_gt


def random_size(img_tensor, gtboxes_and_label, min_ratio=0.8, max_ratio=1.2):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor, gtboxes_and_label = tf.py_func(size, inp=[img_tensor, gtboxes_and_label, min_ratio, max_ratio], Tout=[tf.float32, tf.int32])
    img_tensor = tf.reshape(img_tensor, [h, w, 3])
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])
    return img_tensor, gtboxes_and_label







