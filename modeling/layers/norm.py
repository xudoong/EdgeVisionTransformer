import tensorflow as tf

class LayerNorm(tf.keras.Model):
    def __init__(self, fn, pre=True):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fn = fn
        self.pre = pre

    def call(self, x):
        if self.pre:
            return self.fn(self.norm(x))
        else:
            return self.norm(self.fn(x))