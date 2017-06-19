import tensorflow as tf 


    
class SoftmaxLoss(object):
    
    def __init__(self, max_len, word_dim):
        self.word_dim = word_dim
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, max_len-1), name='labels')

    def cross_entropy(self, logits):
        embed_labels = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[self.word_dim])), self.labels)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = embed_labels)
        loss = tf.reduce_mean(losses, name="loss")
        perplexity = tf.exp(loss)
        tf.summary.scalar('cross_entropy_loss', loss)
        tf.summary.scalar("perplexity", perplexity)
        return loss, perplexity

        
class Optimization(object):

    def adam(self, loss, lr):
        return tf.train.AdamOptimizer(lr).minimize(loss)

   