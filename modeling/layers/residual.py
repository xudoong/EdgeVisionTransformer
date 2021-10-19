import tensorflow as tf

class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x):
        return self.fn(x) + x