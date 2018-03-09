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

class RandomPruner(LenetPruner):
    def __init__(self, logdir, prunedir, data_dir, name="random"):
        LenetPruner.__init__(self, logdir, name + "_" + prunedir, data_dir, name)

    def select_fmap_to_prune(self, sess):
        nidxs = len(self.fmaps_idxs)
        fmap_to_prune = self.fmaps_idxs[random.randint(0, nidxs -1)]
        return fmap_to_prune


