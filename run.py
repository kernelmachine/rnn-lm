import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.data import Data
from lib.display import Display, TensorBoard
from lib.build import BuildLM
import argparse


class Config(object):
    def __init__(self, network, n_train_samples, n_validation_samples, n_epochs, batch_size, embedding_matrix, 
                 max_len, word_dim, train, logdir, save_embedding, save_train_data,
                 calculate_validation):
        self.n_train_samples = n_train_samples
        self.n_validation_samples = n_validation_samples
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.embedding_matrix = embedding_matrix
        self.max_len = max_len
        self.word_dim =word_dim
        self.train = train
        self.logdir = logdir
        self.calculate_validation = calculate_validation
        self.save_embedding = save_embedding
        self.save_train_data = save_train_data
        self.model = lambda x : BuildLM(x, max_len, word_dim) 
        self.build_network = {
                        "lm_stacked_fc": lambda y: self.model.build_lm_stacked_fc(y),
                    }[network]

class Run(object):
        
    def run_lm(self, train_csv, config):
        data = Data()
        display = Display()
        start_token, end_token, word_index = data.run(train_csv,
                 n_train_samples=config.n_train_samples, 
                 n_validation_samples=config.n_validation_samples, 
                 embedding_matrix = config.embedding_matrix,
                 max_len = config.max_len,
                 word_dim=config.word_dim, 
                 train = config.train,
                 save_embedding=config.save_embedding, 
                 save_train_data=config.save_train_data)
        index_to_word = {y : x for x,y in word_index.items()}
        import ipdb; ipdb.set_trace()
        with tf.Graph().as_default() as graph:
           config.model = config.model(data)
           writer = TensorBoard(graph=graph, logdir=config.logdir).writer
           output, loss, perplexity, opt, merged = config.build_network(graph)
           init = tf.global_variables_initializer()
           with tf.Session(graph=graph) as sess:
               sess.run(init)
               for epoch in range(config.n_epochs):
                 train_iter_ = data.batch_generator(config.batch_size)
                 for batch_idx, batch in enumerate(tqdm(train_iter_)):
                    train_batch, train_labels_batch = batch
                    _, batch_train_loss, batch_train_perplexity, _, summary = sess.run([output, loss, perplexity, opt, merged], 
                                                                                    feed_dict={
                                                                                                config.model.network.train : train_batch,
                                                                                                config.model.loss.labels : train_labels_batch
                                                                                              })
                    display.log_train(epoch, batch_idx, batch_train_loss, batch_train_perplexity)

                    if config.calculate_validation:
                        if batch_idx % 100 == 0:
                            batch_valid_perplexity = sess.run([perplexity], feed_dict={
                                                                                config.model.network.train : data.valid,
                                                                                config.model.loss.labels : data.valid_labels
                                                                                })
                            display.log_validation(epoch, batch_idx, batch_valid_perplexity)
                            new_sentence = np.zeros((1,1))
                            new_sentence[0,0] = word_index['start']
                            i = 0
                            while word_index['end'] not in new_sentence:
                                next_word_prob, = sess.run([output], feed_dict = {config.model.network.train : new_sentence})
                                pred = tf.nn.softmax(next_word_prob)
                                next_word = np.argmax(pred.eval())
                                new_sentence = np.concatenate([new_sentence, [[next_word]]], axis=1)
                                print(' '.join([index_to_word[x] for x in new_sentence.tolist()[0]]))
                    writer.add_summary(summary, batch_idx)
                    
        display.done()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Quora-Kaggle',
                                     formatter_class=argparse.
                                     RawTextHelpFormatter)
    parser.add_argument('-e', '--embedding', required=False,
                        help="Supply embedding matrix file")
    parser.add_argument('-s', '--save_embedding', action="store_true",
                        help="Do you want to save embedding to npy file?")
    parser.add_argument('-d', '--master',  required=False,
                        help="Supply master training dataset")
    parser.add_argument('-x', '--train', required=False,
                        help="Supply training data for q1")
    parser.add_argument('-t', '--save_train', action="store_true",
                        help="Do you want to save preprocessed training data to npy file?")
    args = parser.parse_args()
                                         
    config = Config(network="lm_stacked_fc",
                    n_train_samples=298000, 
                    n_validation_samples=10000,
                    n_epochs=10,
                    batch_size=100,
                    embedding_matrix = args.embedding,
                    max_len=10,
                    word_dim=95603, 
                    train = args.train,
                    logdir="/tmp/quora_logs/test", 
                    save_embedding=args.save_embedding,
                    save_train_data=args.save_train,
                    calculate_validation=True)
    Run().run_lm(args.master, config)



## TODO:
## * add word2vec/glove functionality
## * add tensorboard optional
## * make it easier to change dimensions
## * cycle validation set