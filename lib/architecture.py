import tensorflow as tf

class Layer(object):
    def __init__(self, graph, max_len, word_dim):
        self.graph = graph
        self.max_len = max_len
        self.word_dim = word_dim
        self.train = tf.placeholder(dtype=tf.int32, shape=(None, None), name='x')
        self.embedding_matrix = tf.diag(tf.ones(shape=[word_dim]))

    def embed(self, input):
        return tf.nn.embedding_lookup(self.embedding_matrix, input)

    def rnn_temporal_split(self, input):
        num_steps = input.get_shape().as_list()[1]
        embed_split = tf.split(axis=1, num_or_size_splits=num_steps, value=input)
        embed_split = [tf.squeeze(x, axis=[1]) for x in embed_split]
        return embed_split

    def stacked_biRNN(self, input, cell_type, n_layers, network_dim):
        xs = self.rnn_temporal_split(input)
        dropout = lambda y : tf.contrib.rnn.DropoutWrapper(y, output_keep_prob=0.5, seed=42)

        fw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        bw_cells = {"LSTM": [lambda x : tf.contrib.rnn.BasicLSTMCell(x, reuse = None) for _ in range(n_layers)], 
                    "GRU" : [lambda x : tf.contrib.rnn.GRU(x, reuse = None) for _ in range(n_layers)]}[cell_type]
        fw_cells = [dropout(fw_cell(network_dim)) for fw_cell in fw_cells]
        bw_cells = [dropout(bw_cell(network_dim)) for bw_cell in bw_cells]
        fw_stack = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_stack = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, fw_output_state, bw_output_state = tf.contrib.rnn.static_bidirectional_rnn(fw_stack,
                                                                bw_stack,
                                                                xs,
                                                                dtype=tf.float32)
        
        return outputs, fw_output_state, bw_output_state  

    def biRNN(self, input, cell_type, network_dim):
        xs = self.rnn_temporal_split(input)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        bw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse = None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse = None)}[cell_type]
        fw = fw_cell_unit(network_dim)
        bw = bw_cell_unit(network_dim)
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw,
                                                                bw,
                                                                xs,
                                                                dtype=tf.float32)
        return outputs, output_state_fw, output_state_bw

    def rnn(self, input, cell_type, network_dim):
        # xs = self.rnn_temporal_split(input)
        fw_cell_unit = {"GRU": lambda x: tf.contrib.rnn.GRU(x, reuse=None),
                        "LSTM": lambda x: tf.contrib.rnn.BasicLSTMCell(x, reuse=None)}[cell_type]
        fw = fw_cell_unit(network_dim)
        outputs, _ = tf.nn.dynamic_rnn(fw, input, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, network_dim])
        return outputs

    def dense_unit(self, input, name, input_dim, hidden_dim, output_dim):
        bn = tf.nn.batch_normalization(input, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        W1 = tf.get_variable(name="W1_"+name, shape=[input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name="b1_"+name, shape=[hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(bn, W1) + b1)
        d = tf.nn.dropout(h1, keep_prob = 0.5, seed = 42)
        W2 = tf.get_variable(name="W2_"+name, shape=[hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name="b2_"+name, shape=[output_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.matmul(d, W2) + b2
        bn_out = tf.nn.batch_normalization(out, mean = 0.0, variance = 1.0, offset=tf.constant(0.0), scale=None, variance_epsilon=0.001)
        return bn_out

class Network(Layer):
    def __init__(self, graph, max_len, word_dim):
        super(Network, self).__init__(graph, max_len, word_dim)
    

    def lm_stacked_fc_network(self):
        embed = self.embed(self.train)
        with tf.variable_scope("x", reuse=None) as scope:  
            repr = self.rnn(input=embed, cell_type="LSTM", network_dim=512)
        repr_concat = tf.concat(repr, axis=0)
        output = self.dense_unit(repr_concat, "output", 512, 200, self.word_dim)
        return output



