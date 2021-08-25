import tensorflow as tf
from einops.layers.tensorflow import Rearrange


class Attention(tf.keras.Model):

    def __init__(self, dim, num_heads):
        if dim % num_heads != 0:
            raise ValueError(f'hidden_size {dim} must be a multiple of num_heads {num_heads}.')
        
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
 
        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)

        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.num_heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)

        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = self.rearrange_out(out)
        out =  self.to_out(out)
        return out