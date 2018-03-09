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

FLAGS = None
TOTAL_FMAPS = 1120

class LenetPruner:
    def __init__(self, logdir, prunedir, data_dir, name="LenetPruner"):
        self.logdir = logdir
        self.prunedir = prunedir
        self.npruned = 0

        logging.shutdown()
        reload(logging)
        logging.basicConfig(format="[%(process)d]   %(asctime)s %(levelname)s " + name + "| %(message)s")
        self.log = logging.getLogger("train")
        self.log.setLevel(logging.INFO)

        if tf.gfile.Exists(self.logdir + '/' + self.prunedir):
            tf.gfile.DeleteRecursively(self.logdir+ '/' + self.prunedir)
        tf.gfile.MakeDirs(self.logdir + '/' + self.prunedir)

        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        is_training_t = tf.placeholder(tf.bool)

        lenet = Lenet()
        self.lenet = lenet
        y_conv = lenet.inference(x, is_training=is_training_t)

        ####
        # Convulutional layers

        # 32 filters of 5x5
        # Que es la tercera posicion?
        conv1_weights_t = tf.placeholder(tf.float32, (5, 5, 1, 32))
        conv1_biases_t = tf.placeholder(tf.float32, (32))

        #64 filters of 5x5
        conv2_weights_t = tf.placeholder(tf.float32, (5, 5, 32, 64))
        conv2_biases_t = tf.placeholder(tf.float32, (64))

        # fully connected layers
        fc1_weights_t = tf.placeholder(tf.float32, (3136, 1024))
        fc1_biases_t = tf.placeholder(tf.float32, (1024))
        fc2_weights_t = tf.placeholder(tf.float32, (1024, 10))
        fc2_biases_t = tf.placeholder(tf.float32, (10))

        # Numpy arrays with weight & Biases
        self.conv1_weights = np.ones((5, 5, 1, 32), dtype=np.float32)
        self.conv1_biases = np.ones((32), dtype=np.float32)
        self.conv2_weights = np.ones((5, 5, 32, 64), dtype=np.float32)
        self.conv2_biases = np.ones((64), dtype=np.float32)
        self.fc1_weights = np.ones((3136, 1024), dtype=np.float32)
        self.fc1_biases = np.ones((1024), dtype=np.float32)
        self.fc2_weights = np.ones((1024, 10), dtype=np.float32)
        self.fc2_biases = np.ones((10), dtype=np.float32)

        #Initialize with 1 (not prune) all the trained variables
        self.prune_ops = self._prune_op([conv1_weights_t, conv1_biases_t,
                                         conv2_weights_t, conv2_biases_t,
                                         fc1_weights_t, fc1_biases_t,
                                         fc2_weights_t, fc2_biases_t])

        self.loss = lenet.loss_function(y_)
        self.accuracy = lenet.accuracy(y_)
        lenet.add_summaries()

        self.last_pruned_t = tf.Variable(0,dtype=tf.int32, name="last_pruned", trainable=False)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('pruned', self.last_pruned_t)
        
        prune_feed_dict = {}
        prune_feed_dict[conv1_weights_t] = self.conv1_weights
        prune_feed_dict[conv1_biases_t] = self.conv1_biases
        prune_feed_dict[conv2_weights_t] = self.conv2_weights
        prune_feed_dict[conv2_biases_t] = self.conv2_biases
        prune_feed_dict[fc1_weights_t] = self.fc1_weights
        prune_feed_dict[fc1_biases_t] = self.fc1_biases
        prune_feed_dict[fc2_weights_t] = self.fc2_weights
        prune_feed_dict[fc2_biases_t] = self.fc2_biases

        self.feed_dict = FeedDict(data_dir, x, y_, is_training_t, extra_dict=prune_feed_dict)

    def _prune_op(self, fmaps_to_prune):
        """ Here we set to '0' those fmaps to be pruned
        
        Args:
            fmaps_to_prune: Tensors of weights/biases with 0-1 values.
                            1 if that weight must remain, 0 if has to be
                            pruned. The order of the variables is the same
                            as in 'tf.GraphKeys.TRAINABLE_VARIABLES'.

        Returns:
            An operation once executed will prune selected weights.
        """
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        prune_ops = []
        
        for i in range(len(variables)):
            prune_ops.append(tf.assign(variables[i], variables[i] * fmaps_to_prune[i]))

        return prune_ops

    def has_to_stop(self, fmaps_to_prune, total_fmaps):
        acc = np.sum(self.conv1_biases)
        acc += np.sum(self.conv2_biases)
        acc += np.sum(self.fc1_biases)
        return acc == total_fmaps - fmaps_to_prune

    def prune_fmap(self, fmap):
        if fmap < 32:
            self.conv1_weights[:,:,0,fmap] = 0.0
            self.conv1_biases[fmap] = 0.0

            self.conv2_weights[:,:,fmap, :] = 0.0
        elif fmap < 96:
            fmap -= 32
            self.conv2_weights[:,:,:,fmap] = 0.0
            self.conv2_biases[fmap] = 0.0
            
            for i in xrange(49):
                self.fc1_weights[fmap + i * 64, :] = 0.0
        else:
            fmap -= 96
            self.fc1_weights[:,fmap] = 0.0
            self.fc1_biases[fmap] = 0.0

            self.fc2_weights[fmap, :] = 0.0

    def fmap_already_pruned(self, fmap):
        if fmap < 32:
            if self.conv1_biases[fmap] == 0.0:
                return True
        elif fmap < 96:
            if self.conv2_biases[fmap - 32] == 0.0:
                return True
        else:
            if self.fc1_biases[fmap - 96] == 0.0:
                return True
        return False

    def select_fmap_to_prune(self, sess):
        """
        Virtual abstract function that returns the indx of the fmap
        to prune. 
        The idxs of the fmaps to prune are stored in the attribute
        'self.fmaps_idxs'. This is a list of prunnable fmaps idxs, for 
        simplicity for this test, these indexes are every fmap in the net, from
        0 to 1120, being the first 32 idxs corresponding to the 32 fmaps of the 
        first layer, the idxs [32, 64) the idxs of the second layer and the last
        1024 the ones from the first fully connected layer. The second fully 
        connected is not pruned. 

        Returns:
            The index of the fmap to be pruned.
        """
        pass
                
    def prune_attempt(self, sess):
        fmap_to_prune = self.select_fmap_to_prune(sess)
        #log.info("Pruning: %d" % fmap_to_prune)
        if self.fmap_already_pruned(fmap_to_prune):
            return False

        sess.run(tf.assign(self.last_pruned_t, fmap_to_prune))
        self.prune_fmap(fmap_to_prune)

        return True

    def prune(self, fmaps_to_prune=0, layer_to_prune="all"):
        fmaps_to_prune = fmaps_to_prune if fmaps_to_prune != 0 else TOTAL_FMAPS
        self.npruned = 0
        # To make experiments repeteable.
        random.seed(a=0)
        
        prunable_layers = {
            "conv1": [0, 32, 32],
            "conv2": [32, 96, 64],
            "fc1": [96, 1120, 1024]
        }

        minfmap, maxfmap, totalfmaps = prunable_layers.get(layer_to_prune, [0, 1120, 1120])
        fmaps_to_prune = min(fmaps_to_prune, totalfmaps)
        self.fmaps_idxs = np.arange(totalfmaps)
        self.fmaps_idxs = self.fmaps_idxs + minfmap
        self.fmaps_idxs = self.fmaps_idxs.tolist()

        merged = tf.summary.merge_all()

        prune_writer = tf.summary.FileWriter(self.logdir + '/' + self.prunedir)
        prune_writer.add_graph(tf.get_default_graph())
        
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        test_feed_dict = self.feed_dict.test()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            checkpoint_path = tf.train.latest_checkpoint(self.logdir)
            saver.restore(sess, checkpoint_path)
            self.log.info('Loading graph from: %s' % checkpoint_path)
            while not self.has_to_stop(fmaps_to_prune, TOTAL_FMAPS):
                sess.run(self.prune_ops, feed_dict=test_feed_dict)

                summary, acc = sess.run([merged, self.accuracy], feed_dict=test_feed_dict)
                prune_writer.add_summary(summary, self.npruned)
                self.log.info('Test accuracy %.8f, pruned: %d' % (acc,  self.npruned))
                ok = False
                while not ok:
                    ok = self.prune_attempt(sess)
                self.npruned += 1
            sess.run(self.prune_ops, feed_dict=test_feed_dict)
            summary, acc = sess.run([merged, self.accuracy], feed_dict=test_feed_dict)
            prune_writer.add_summary(summary, self.npruned)
            self.log.info('Test accuracy %.8f, pruned: %d' % (sess.run(self.accuracy, feed_dict=test_feed_dict),  self.npruned))
            #saver.save(sess, self.logdir + '/prune/', global_step=step)
            prune_writer.close()