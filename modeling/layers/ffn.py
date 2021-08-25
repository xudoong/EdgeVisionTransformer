import tensorflow as tf
from .activation import gelu


class FeedForward(tf.keras.Model):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation=gelu),
                                        tf.keras.layers.Dense(dim)])

    def call(self, x):
        return self.net(x)