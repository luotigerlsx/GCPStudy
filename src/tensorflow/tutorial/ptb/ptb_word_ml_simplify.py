import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
import reader_simplify

BASIC = "basic"
BLOCK = "block"

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", '/Users/luoshixin/Downloads/data/simple-examples/data',
                    "Where the training/test data is stored.")

FLAGS = flags.FLAGS


class PTBConfig:
    """Small config."""

    def __init__(self,
                 init_scale=0.1,
                 learning_rate=1.0,
                 max_grad_norm=5,
                 num_layers=2,
                 num_steps=20,
                 hidden_size=200,
                 embedding_size=200,
                 max_epoch=4,
                 max_max_epoch=13,
                 keep_prob=1.0,
                 batch_size=20,
                 vocab_size=10000,
                 rnn_mode=BLOCK):
        self.init_scale = init_scale
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.max_epoch = max_epoch
        self.max_max_epoch = max_max_epoch
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.rnn_mode = rnn_mode


class PTBModel:

    def __init__(self, config: PTBConfig):
        self.config = config

        '''Basic configurations
        '''
        self.batch_size = self.config.batch_size
        self.time_steps = self.config.num_steps

        self.vocab_size = self.config.vocab_size
        self.embedding_size = self.config.embedding_size

        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.keep_prob = self.config.keep_prob
        self.max_grad_norm = self.config.max_grad_norm

        '''Pre-define some important handlers
        '''
        self._session = None
        self._is_training = None
        self._training_op = None
        self._graph = None

    def _get_rnn_cell(self):
        with tf.variable_scope('rnn_cell'):
            lstm_cell = rnn.LSTMBlockCell(self.hidden_size, forget_bias=1.0)
            return rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=self._keep_prob)

    def _build_rnn_graph(self, inputs):
        with tf.variable_scope('rnn_graph'):
            stack_cell = rnn.MultiRNNCell(
                cells=[self._get_rnn_cell() for i in range(self.num_layers)],
                state_is_tuple=True
            )

            rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(
                cell=stack_cell,
                output_size=self.vocab_size
            )

            # inputs should have size [batch, time_steps, num_inputs]
            outputs, state = tf.nn.dynamic_rnn(
                cell=rnn_cell,
                inputs=inputs,
                dtype=tf.float32
            )

            return outputs, state

    def _build_graph(self):
        self._is_training = tf.placeholder_with_default(input=True, shape=(), name='is_training')
        self._keep_prob = tf.placeholder_with_default(self.keep_prob, shape=(), name='keep_probability')
        # [batch_size, time_steps]
        X = tf.placeholder(shape=[None, self.time_steps], dtype=tf.int32)
        y = tf.placeholder(shape=[None, self.time_steps], dtype=tf.int32)

        with tf.device("/cpu:0"):
            # [vocab_size, embedding_size]
            embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)
            # [batch_size, time_steps, embedding_size]
            inputs = tf.nn.embedding_lookup(embedding, X)

        inputs = tf.nn.dropout(inputs, self._keep_prob)
        # [batch_size, time_steps, vocab_size]
        outputs, final_state = self._build_rnn_graph(inputs)

        # Use the contrib sequence loss and average over the batches
        # [batch_size]
        loss = seq2seq.sequence_loss(
            outputs,
            y,
            tf.ones_like(y, dtype=tf.float32),
            # tf.ones([self.batch_size, self.time_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        cost = tf.reduce_sum(loss)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        '''Make important handler of the graph available
        '''

        self._X, self._Y = X, y
        self._training_op, self._cost = train_op, cost
        self._init, self._saver = init, saver
        self._final_state = final_state

    def close_session(self):
        if self._session:
            self._session.clost()

    def fit(self, train_data, valid_data=None, n_epoches=5):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        best_cost = 1e10
        check_interval = 100

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            sess.run(self._init)

            for epoch in range(n_epoches):

                train_data_iter = reader_simplify.ptb_producer(train_data,
                                                               self.batch_size,
                                                               self.time_steps)

                for X_batch, y_batch in train_data_iter:
                    feed_dict = {self._X: X_batch, self._Y: y_batch}

                    if self._is_training is not None:
                        feed_dict[self._is_training] = True
                        feed_dict[self._keep_prob] = self.keep_prob

                    _, cost = sess.run([self._training_op, self._cost], feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)

                    if cost < best_cost:
                        best_cost = cost
                        self._saver.save(sess, './tmp/')
                        print('Current best cost is: {}'.format(best_cost))
                        # if valid_data:
                        #     X_valid, y_valid = next(reader_simplify
                        #                             .ptb_producer(valid_data,
                        #                                           len(valid_data) // (self.time_steps * 2),
                        #                                           self.time_steps))
                        #     valid_cost = sess.run(self._cost, feed_dict={self._X: X_valid,
                        #                                                  self._Y: y_valid,
                        #                                                  self._is_training: False,
                        #                                                  self._keep_prob: 1.0})
                        #     print('The validation cost is: {}'.format(valid_cost))


def main():
    config = PTBConfig()
    raw_data = reader_simplify.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data
    model = PTBModel(config)
    model.fit(train_data, valid_data)


if __name__ == '__main__':
    main()
