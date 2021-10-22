import numpy as np
import tensorflow as tf
import math
from .residual import Residual
from .norm import LayerNorm
from .attention import Attention
from .ffn import FeedForward

class TransformerEncoderBlock(tf.keras.Model):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size, norm_first=False):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                LayerNorm(Residual(Attention(hidden_size, num_heads=num_heads)), pre=norm_first),
                LayerNorm(Residual(FeedForward(hidden_size, intermediate_size)), pre=norm_first)
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


class  TransformerEncoderBlock_Pruned(tf.keras.Model):
    def __init__(self, hidden_size, num_layers, num_remain_heads_list, intermediate_size_list, head_size=64, norm_first=False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                LayerNorm(Residual(Attention(hidden_size, num_heads=num_remain_heads_list[i], h_k=head_size)), pre=norm_first),
                LayerNorm(Residual(FeedForward(hidden_size, intermediate_size_list[i])), pre=norm_first)
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)


class TokenPerformer(tf.keras.Model):
    '''
    T2T-Module performer for T2T-ViT
    '''
    def __init__(self, head_size, num_heads, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.hidden_size = head_size * num_heads
        self.kqv = tf.keras.layers.Dense(self.hidden_size * 3)
        self.dp = tf.keras.layers.Dropout(dp1)
        self.attn_output = tf.keras.layers.Dense(self.hidden_size)
        self.num_heads = num_heads
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.epsilon = 1e-8  # for stable in division   

        self.mlp = tf.keras.Sequential([
            FeedForward(self.hidden_size, self.hidden_size),
            tf.keras.layers.Dropout(dp2)
        ])

        self.m = int(self.hidden_size * kernel_ratio)
        self.w = self.add_weight('w',
                                 shape=[self.m, self.hidden_size],
                                 initializer=tf.keras.initializers.Orthogonal(),
                                 dtype=tf.float32,
                                 trainable=False)
        self.w = self.w * math.sqrt(self.m)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = tf.math.reduce_sum(x * x, axis=-1, keepdims=True)
        broadcast_shape = tf.where([True, True, False], tf.shape(xd), [0, 0, self.m])
        xd = tf.broadcast_to(xd, broadcast_shape) / 2
        wtd = tf.einsum('bti,mi->btm', tf.convert_to_tensor(x, dtype=tf.float32), self.w)

        return tf.exp(wtd - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = tf.split(self.kqv(x), 3, axis=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q) # (B, T, m), (B, T, m)
        D = tf.einsum('bti,bi->bt', qp, tf.math.reduce_sum(kp, axis=1)) # (B, T, m) * (B, m) -> (B, T, 1)
        D = tf.expand_dims(D, axis=2)
        kptv = tf.einsum('bin,bim->bnm', tf.convert_to_tensor(v, dtype=tf.float32), kp) # (B, emb, m)
        broadcast_shape = tf.where([True, True, False], tf.shape(D), [0, 0, self.hidden_size])
        y = tf.einsum('bti,bni->btn', qp, kptv) / (tf.broadcast_to(D, broadcast_shape) + self.epsilon) # (B, T, emb) / Diag
        
        # skip connection
        y = v + self.dp(self.attn_output(y))
        return y

    def call(self, x):
        x = self.norm1(x)
        x = self.single_attn(x)
        x = x + self.mlp(self.norm2(x))

        return x
