#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  read_binary_model.py

"""

__author__ = 'Welkin'
__date__ = '2019-03-13 17:54'

import tensorflow as tf

GraphDef = tf.GraphDef()


def read_binary_model(filename):
    with open(filename, 'rb') as f:
        graph_def = GraphDef.FromString(f.read())
        operation = tf.import_graph_def(graph_def)
        print("read into binary graph: ", operation)
        tf.train.write_graph(graph_def, '../inception-2015-12-05/', 'inception_v3.pbtxt', as_text = True)


read_binary_model('../inception-2015-12-05/classify_image_graph_def.pb')
