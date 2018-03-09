from layers import *

class Lenet:
    def __init__(self, name="Lenet"):
        self.flops = 0
        self.name="Lenet"

    def inference(self, input_image, is_training, width=[32, 64, 1024]):
        with tf.name_scope(self.name):
            self._inference(input_image, is_training, width)

    def _inference(self, input_image, is_training, width=[32, 64, 1024]):
        """deepnn builds the graph for a deep net for classifying digits.

        Args:
            input_image: an input tensor with the dimensions (N_examples, 784),
            where 784 is the number of pixels in a standard MNIST image.

        Returns:
            A tensor y. y is a tensor of shape (N_examples, 10), with values
            equal to the logits of classifying the digit into one of 10 classes (the
            digits 0-9).
        """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale
        with tf.name_scope('reshape'):
            x_image = tf.reshape(input_image, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        h_conv1 = conv(x_image, width[0], 5, is_training=is_training, scope='conv1')
        self.h_conv1 = h_conv1
        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool2d(h_conv1, 'pool1')

        # Second convolutional layer -- maps 32 feature maps to 64.
        h_conv2 = conv(h_pool1, width[1], 5, is_training=is_training, scope='conv2')
        
        # Second pooling layer.
        h_pool2 = max_pool2d(h_conv2, 'pool2')
            
        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        h_pool2_reshaped = tf.reshape(h_pool2, [-1, 7*7*width[1]])
        h_fc1 = fc(h_pool2_reshaped, width[2], is_training=is_training, scope='fc1')
        self.h_fc1 = h_fc1
        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        h_fc1_drop = tf.contrib.layers.dropout(h_fc1, is_training=is_training, scope='dropout')

        # Map the 1024 features to 10 classes, one for each digit
        y_conv = fc(h_fc1_drop, 10, is_training=is_training, scope='fc2', activation_fn=None)
        self.prediction = y_conv
        return y_conv

    def loss_function(self, target):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=self.prediction)
            # We use sum to avoid underflow when prunning with taylor criterion.
            cross_entropy = tf.reduce_sum(cross_entropy)
        return cross_entropy

    def accuracy(self, target):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(target, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
        return accuracy

    def add_summaries(self):
        """Attach summaries for all trainable variables."""
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
        for var in variables:
            self._add_variable_summaries(var)

    def _add_variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(var.name[0:-3]):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)