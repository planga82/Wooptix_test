from tensorflow.examples.tutorials.mnist import input_data

class FeedDict:
    def __init__(self, data_dir, x, y_, is_training_t, batch_size=100, extra_dict={}):
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        self.batch_size = batch_size
        self.is_training_t = is_training_t
        self.x = x
        self.y_ = y_

        self.test_feed_dict = {}
        self.test_feed_dict[self.is_training_t] = False
        self.test_feed_dict[self.x] = self.mnist.validation.images
        self.test_feed_dict[self.y_] = self.mnist.validation.labels
        self.test_feed_dict.update(extra_dict)

        self.train_feed_dict = {}
        self.train_feed_dict[self.is_training_t] = True
        train_batch = self.mnist.train.next_batch(batch_size)
        self.train_feed_dict[self.x] = train_batch[0]
        self.train_feed_dict[self.y_] = train_batch[1]
        self.train_feed_dict.update(extra_dict)

    def test(self): 
        return self.test_feed_dict

    def train(self):
        return self.train_feed_dict

    def next_batch(self):
        train_batch = self.mnist.train.next_batch(self.batch_size)
        self.train_feed_dict[self.x] = train_batch[0]
        self.train_feed_dict[self.y_] = train_batch[1]