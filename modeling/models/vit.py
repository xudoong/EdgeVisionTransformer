import tensorflow as tf

from einops.layers.tensorflow import Rearrange
from layers.transformer_encoder import TransformerEncoderBlock
from layers.activation import gelu


class ViT(tf.keras.Model):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = self.add_weight("position_embeddings",
                                             shape=[num_patches + 1,
                                                    dim],
                                             initializer=tf.keras.initializers.RandomNormal(),
                                             dtype=tf.float32)
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.cls_token = self.add_weight("cls_token",
                                         shape=[1,
                                                1,
                                                dim],
                                         initializer=tf.keras.initializers.RandomNormal(),
                                         dtype=tf.float32)

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        self.transformer = TransformerEncoderBlock(dim, depth, heads, mlp_dim)
 
        self.to_cls_token = tf.identity

        self.mlp_head = tf.keras.Sequential([tf.keras.layers.Dense(mlp_dim, activation=gelu),
                                        tf.keras.layers.Dense(num_classes)])

    @tf.function
    def call(self, img):
        shapes = tf.shape(img)

        x = self.rearrange(img) # [b, h * w, p * p * c]
        x = self.patch_to_embedding(x) # [b, h * w = n, dim]

        cls_tokens = tf.broadcast_to(self.cls_token,(shapes[0], 1, self.dim)) # [b, 1, dim]
        x = tf.concat((cls_tokens, x), axis=1) # [b, n + 1, dim]
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0]) # [b, dim]
        return self.mlp_head(x) # [b, num_classes]