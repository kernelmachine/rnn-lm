import tensorflow as tf
import numpy as np
import pandas as pd 
from keras.preprocessing import sequence, text
import string
from gensim.models import KeyedVectors

class Data(object):
    def __init__(self):
        self.train_csv = None
        self.train = None
        self.train_labels = None
        self.valid = None
        self.valid_labels = None
        self.embedding_matrix = None

    def import_data(self, train_csv):
        print("importing data...")
        df = pd.read_csv(train_csv)
        df = df.dropna(axis=0)
        return df

    def preprocess_char(self, df):
        print("preprocessing data...")
        import string
        df['sentences'] = df.sentences.apply(lambda x: '^' + x + '$')
        vocab_chars = string.ascii_lowercase + '^$0123456789 '
        vocab2ix_dict = {char:ix for ix, char in enumerate(vocab_chars)}
        vocab_length = len(vocab_chars) + 1
        start_token = vocab2ix_dict['^']
        end_token = vocab2ix_dict['$']
        def sentence2onehot(sentence, vocab2ix_dict = vocab2ix_dict):
            # translate sentence string into indices
            sentence_ix = [vocab2ix_dict[x] for x in list(sentence) if x in vocab_chars]
            # Pad or crop to embedding dimension
            sentence_ix = (sentence_ix + [0]*self.max_len)[0:self.max_len]
            return(sentence_ix)
        one_hots = df.sentences.str.lower().apply(sentence2onehot)
        one_hots = np.matrix(one_hots.tolist())
        self.train = one_hots[:, :-1]
        self.train_labels = one_hots[:, 1:]
        if self.embedding_matrix is None:
            self.embedding_matrix = tf.diag(tf.ones(shape=[self.word_dim]))
        print("example sentence: %s" % df['sentences'][0])
        print("x: %s" % self.train[0,:])
        print("y: %s" % self.train_labels[0,:])
        return start_token, end_token, vocab2ix_dict
    
    def preprocess_word2vec(self, df, save_embedding=False, save_train_data=False):
        print("preprocessing data...")
        df['sentences'] = df.sentences.apply(lambda x: 'start ' + x + ' end')
        tk = text.Tokenizer(num_words=self.word_dim)
        tk.fit_on_texts(list(df.sentences.values.astype(str)))
        word_index = tk.word_index
        train = tk.texts_to_sequences(df.sentences.values)
        sequences = sequence.pad_sequences(train, maxlen=self.max_len)
        self.train = sequences[:, :-1]
        self.train_labels = sequences[:, 1:]
        start_token = word_index['start']
        end_token = word_index['end']
        if save_train_data:
            print("saving preprocessed training data...")
            np.save("%s.npy" % self.train_csv, self.train)
        if self.embedding_matrix is None:
            self.embedding_matrix = tf.diag(tf.ones(shape=[self.word_dim]))
        print("example sentence: %s" % df['sentences'][0])
        print("x: %s" % self.train[0,:])
        print("y: %s" % self.train_labels[0,:])
        
        return start_token, end_token, word_index

    def subsample(self, n_train_samples, n_validation_samples):
        print("subsampling data...")
        train_size = self.train.shape[0]
        global_idx = np.random.choice(train_size, n_train_samples + n_validation_samples, replace=False)
        np.random.shuffle(global_idx)
        train_sample_idx = global_idx[:n_train_samples]
        validation_sample_idx = global_idx[n_train_samples:]
        self.valid = self.train[validation_sample_idx, :]
        self.train = self.train[train_sample_idx, :]
        self.valid_labels = self.train_labels[validation_sample_idx,:]
        self.train_labels = self.train_labels[train_sample_idx, :]

        
        
    def batch_generator(self, batch_size):
            l = self.train.shape[0]
            for ndx in range(0, l, batch_size):
                yield (self.train[ndx:min(ndx + batch_size, l), :],
                    self.train_labels[ndx:min(ndx + batch_size, l),:],
                    )
                        
    def run(self, train_csv, n_train_samples=400000, n_validation_samples=10000, embedding_matrix=None, max_len=50, word_dim=8000, train=None,save_embedding=False, save_train_data=False):
        self.train_csv = train_csv
        df = self.import_data(train_csv)
        if embedding_matrix is not None:
            print("loading embedding matrix from %s" % embedding_matrix)
            self.embedding_matrix = np.load(embedding_matrix)
        self.word_dim = word_dim
        self.max_len = max_len
        if train is not None:
            print("loading train_x1 from %s" % train)
            self.train = np.load(train)
        start_token, end_token, word_index = self.preprocess_char(df)
        self.subsample(n_train_samples, n_validation_samples)
        return start_token, end_token, word_index