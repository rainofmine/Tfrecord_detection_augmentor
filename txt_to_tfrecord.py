# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import glob
import cv2
import config as cfg
from label_dict import *
from tools import *

tf.app.flags.DEFINE_string('type', 'train', 'train or val')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_txt(txt_path):
    box_list = []
    txt_file = open(txt_path, 'r')
    for line in txt_file.readlines():
        tmp_box = line.strip().split(',')
        tmp_box[-1] = NAME_LABEL_MAP[tmp_box[-1]]
        box_list.append(tmp_box)
    gtbox_label = np.array(box_list, dtype=np.int32)
    return gtbox_label

def convert_txt_to_tfrecord():
    type = FLAGS.type
    data_dir = os.path.join(cfg.ROOT_PATH, 'Dataset_' + type)
    txt_path = os.path.join(data_dir, 'Annotations')
    image_path = os.path.join(data_dir, 'JPEGImages')
    save_path = os.path.join(cfg.ROOT_PATH, cfg.TFRECORD_PATH, cfg.DATASET_NAME + '_' + type + '.tfrecord')
    mkdir(FLAGS.save_dir)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)

    for count, txt in enumerate(glob.glob(txt_path + '/*.txt')):
        # to avoid path error in different development platform
        txt = txt.replace('\\', '/')

        img_name = txt.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name
        print(img_name)

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        gtbox_label = read_txt(txt)

        img = cv2.imread(img_path)
        img_height, img_width = img.shape[0], img.shape[1]

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            # 'img_name': _bytes_feature(img_name.encode()),
            'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(txt_path + '/*.txt')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    convert_txt_to_tfrecord()
