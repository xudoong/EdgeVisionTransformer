from numpy.core import numeric
import tensorflow as tf

from einops.layers.tensorflow import Rearrange
from modeling.layers.transformer_encoder import TransformerEncoderBlock, TransformerEncoderBlock_Pruned
from modeling.layers.activation import gelu


class ViT(tf.keras.Model):

    def __init__(self, *, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072):
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

        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        self.transformer = TransformerEncoderBlock(dim, depth, heads, mlp_dim)

        self.to_cls_token = tf.identity

        self.mlp_head = tf.keras.Sequential([tf.keras.layers.Dense(mlp_dim, activation=gelu),
                                             tf.keras.layers.Dense(num_classes)])

    @tf.function
    def call(self, img):
        shapes = tf.shape(img)

        x = self.rearrange(img)  # [b, h * w, p * p * c]
        x = self.patch_to_embedding(x)  # [b, h * w = n, dim]

        cls_tokens = tf.broadcast_to(
            self.cls_token, (shapes[0], 1, self.dim))  # [b, 1, dim]
        x = tf.concat((cls_tokens, x), axis=1)  # [b, n + 1, dim]
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])  # [b, dim]
        return self.mlp_head(x)  # [b, num_classes]


class ViT_Pruned(ViT):

    def __init__(self, *, image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, head_size=64, prune_encoding='all_head12_ffn1.0'):
        prune_setting, num_remain_heads, ffn_thresholds = self.decode_prune_encoding(prune_encoding)
        if prune_setting == 'all':
            num_remain_heads_list = [num_remain_heads for _ in range(depth)]
            intermediate_size_list = [int(ffn_thresholds * mlp_dim) for _ in range(depth)]
        else: # prune_setting == 'layerwise'
            assert(len(num_remain_heads) == depth and len(ffn_thresholds) == depth)
            num_remain_heads_list = num_remain_heads
            intermediate_size_list = [int(ffn_thresholds[i] * mlp_dim) for i in range(depth)]

        super().__init__(image_size=image_size, patch_size=patch_size,
                         num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)

        # override TransformerEncoderBlock
        self.transformer = TransformerEncoderBlock_Pruned(hidden_size=dim, num_layers=depth, num_remain_heads_list=num_remain_heads_list, 
                                                          intermediate_size_list=intermediate_size_list, head_size=head_size, norm_first=True)
        
    def decode_prune_encoding(self, prune_encoding: str):
        tokens = prune_encoding.split('_')
        print(tokens)
        prune_setting = tokens[0]
        assert prune_setting in ['layerwise', 'all']
        if prune_setting == 'all':
            # e.g. prune_encoding = 'all_head12_ffn1.0': all layers remain 12 heads and 100% ffn
            head_setting = tokens[1]
            ffn_setting = tokens[2]
            num_heads = int(head_setting.replace('head', ''))
            ffn_threshold = float(ffn_setting.replace('ffn', ''))
            return prune_setting, num_heads, ffn_threshold
        else: # prune_setting == 'layerwise'
            # e.g. prune_encoding = 'layerwise_h2-d1.0_h3-d0.5_h1-d0.5'
            num_heads_list = []
            ffn_threshold_list = []
            for token in tokens[1: ]:
                hx, dx = token.split('-')
                num_heads_list.append(int(hx.replace('h', '')))
                ffn_threshold_list.append(float(dx.replace('d', '')))
            return prune_setting, num_heads_list, ffn_threshold_list


def get_deit_base():
    return ViT(dim=768, depth=12)


def get_deit_small():
    return ViT(dim=384, heads=6, mlp_dim=384 * 4)


def get_deit_tiny():
    return ViT(dim=192, heads=3, mlp_dim=192 * 4)
