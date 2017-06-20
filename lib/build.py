import tensorflow as tf
from lib.architecture import Network
from lib.optimization import *

class BuildLM(object):
    def __init__(self, graph, max_len, word_dim):
        self.network = Network(graph, max_len=max_len, word_dim=word_dim)
        self.loss = SoftmaxLoss(max_len=max_len, word_dim=word_dim)
        self.opt = Optimization()
    
    def build_lm_stacked_fc(self, graph):
        print("building lm_stacked_fc network...")
        output = self.network.lm_stacked_fc_network()
        loss, perplexity = self.loss.cross_entropy(output)
        opt = self.opt.adam(loss, 0.002)
        # acc, train_summ, valid_summ = self.accuracy.sigmoid_accuracy(self.loss.labels, output)
        merged = tf.summary.merge_all()
        return output, loss, perplexity, opt, merged

    