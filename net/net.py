from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

from utils.img_utils import *


def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Conv2D(128, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Conv2D(256, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.Conv2D(256, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1]),
            tf.keras.layers.Conv2D(512, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(512, [3, 3], padding="same", activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1]),
            tf.keras.layers.Conv2D(512, [2, 2], strides=[2, 1], padding="same", activation='relu'),
            tf.keras.layers.Reshape((25, 512))
        ])

        self.gru = gru(self.enc_units)

    def call(self, x):
        x = self.cnn(x)
        output, state = self.gru(x)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(enc_output, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x1 = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x2 = tf.concat([tf.expand_dims(context_vector, 1), x1], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x2)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights
