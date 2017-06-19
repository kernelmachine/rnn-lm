import tensorflow as tf
from tqdm import tqdm

class Display(object):
    def log_train(self, epoch, batch_idx, batch_train_loss, batch_train_perplexity):
        tqdm.write("EPOCH: %s, BATCH: %s, TRAIN LOSS: %s, TRAIN PERPLEXITY: %s" % (epoch, batch_idx, batch_train_loss, batch_train_perplexity))
    def log_validation(self, epoch, batch_idx, batch_valid_perplexity):
        tqdm.write("EPOCH: %s, BATCH: %s, VALIDATION PERPLEXITY: %s" % (epoch, batch_idx,  batch_valid_perplexity))
    def done(self):
        print("Done!")

class TensorBoard(object):
    def __init__(self, graph, logdir):
        self.writer = tf.summary.FileWriter(logdir,graph)
        self.logdir = logdir

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)