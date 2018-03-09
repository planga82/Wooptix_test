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
from RandomPruner import RandomPruner
from MinWeightPruner import MinWeightPruner

FLAGS = None

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def main(_):
    random_pruner = RandomPruner(FLAGS.logdir, FLAGS.layer_to_prune, FLAGS.data_dir, name="random")
    random_pruner.prune(layer_to_prune=FLAGS.layer_to_prune)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                                            default='/tmp/tensorflow/mnist/input_data',
                                            help='Directory for storing input data')
    parser.add_argument('--logdir', type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                                     'tensorflow/mnist/logs/mnistoo'),
            help='Summaries log directory')
    parser.add_argument('--layer_to_prune', type=str,
            default="all",
            help='Layer to prune')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
