import tensorflow as tf

from .residual import Residual
from .norm import LayerNorm
from .attention import Attention
from .ffn import FeedForward

class TransformerEncoderBlock(tf.keras.Model):

    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                LayerNorm(Residual(Attention(hidden_size, num_heads=num_heads))),
                LayerNorm(Residual(FeedForward(hidden_size, intermediate_size)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)