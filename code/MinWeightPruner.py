from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import tensorflow as tf
import os
import logging
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
from Lenet import Lenet
from FeedDict import FeedDict
from LenetPruner import LenetPruner

FLAGS = None
TOTAL_FMAPS = 1120

class MinWeightPruner(LenetPruner):
    def __init__(self, logdir, prunedir, data_dir, name="Min weight"):
        LenetPruner.__init__(self, logdir, "min_weight_" + prunedir, data_dir, name)
        weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
        biases = tf.get_collection(tf.GraphKeys.BIASES)

        reduced = []

        # Conv1:
        reduced.append(tf.reduce_sum(tf.abs(weights[0]), [0,1,2]))
        # Conv2:
        reduced.append(tf.reduce_sum(tf.abs(weights[1]), [0,1,2]))
        # FC1:
        reduced.append(tf.reduce_sum(tf.abs(weights[2]), [0]))

        self.conv1_red = reduced[0]
        self.conv2_red = reduced[1]
        self.fc1_red = reduced[2]

        tf.summary.histogram('conv1_red', self.conv1_red)
        tf.summary.histogram('conv2_red', self.conv2_red)
        tf.summary.histogram('fc1_red', self.fc1_red)

    def select_fmap_to_prune(self, sess):

        """ Here we choose the map with the min value"""
        # Run
        red = sess.run([self.conv1_red, self.conv2_red, self.fc1_red], feed_dict=self.feed_dict.test())

        # Join all the results in one list and order it(index order)
        red = np.array([red[0].tolist()+ red[1].tolist()+ red[2].tolist()])[0]
        argsorted = np.argsort(red)
        
        for i in xrange(len(argsorted)):
            idx = argsorted[i]
            if idx in self.fmaps_idxs:
                self.fmaps_idxs.remove(idx)
                return idx
        raise "Can't prune any fmap."


