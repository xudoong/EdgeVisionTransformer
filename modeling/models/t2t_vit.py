import tensorflow as tf
import numpy as np
from modeling.layers.transformer_encoder import TokenPerformer, TransformerEncoderBlock
from modeling.layers.embedding import get_sinusoid_encoding


class tf_Unfold(tf.keras.Model):
    '''
    tensorflow implementation of torch.nn.Unfold
    expect input image to be channel-last
    '''
    def __init__(self, kernel_size, stride, padding, channel_last=False, exact_same_as_torch=False):
        super().__init__()
        self.kernel_sizes = [1, kernel_size, kernel_size, 1]
        self.strides = [1, stride, stride, 1]
        self.paddings = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]])
        self.channel_last = channel_last
        self.exact_same_as_torch = exact_same_as_torch
        
    def call(self, x):
        x = tf.pad(x, self.paddings)

        if self.exact_same_as_torch:
            x = [tf.image.extract_patches(
                    x[:, :, :, i: i + 1],
                    sizes=self.kernel_sizes,
                    strides=self.strides,
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                ) for i in range(x.shape[3])]
            x = tf.concat(x, axis=3)
        else:
            x = tf.image.extract_patches(
                x, self.kernel_sizes, self.strides, [1, 1, 1, 1], 'VALID'
            )

        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])
        if not self.channel_last:
            x = tf.transpose(x, [0, 2, 1])
        return x


class T2T_module(tf.keras.Model):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, image_size=224, tokens_type='performer', in_channels=3, embedding_size=768, token_size=64):
        super().__init__()
        if tokens_type == 'performer':
            self.soft_split0 = tf_Unfold(kernel_size=7, stride=4, padding=2, channel_last=True)
            self.soft_split1 = tf_Unfold(kernel_size=3, stride=2, padding=1, channel_last=True)
            self.soft_split2 = tf_Unfold(kernel_size=3, stride=2, padding=1, channel_last=True)

            self.performer1 = TokenPerformer(head_size=token_size, num_heads=1, kernel_ratio=0.5)
            self.performer2 = TokenPerformer(head_size=token_size, num_heads=1, kernel_ratio=0.5)
            self.project = tf.keras.layers.Dense(embedding_size)

        else:
            raise NotImplementedError('T2T_module with token_type other than performer is not supported')

        self.num_patches = (image_size // (4 * 2 * 2)) * (image_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def call(self, x):
        # step0: soft split
        # input shape: [B, 224, 224, 3]
        x = self.soft_split0(x) # [B, 56x56, 7x7x3]
        # iteration1: re-structurization/reconstruction
        x = self.performer1(x) #[B, 56x56, 64]
        B, new_HW, C = x.shape
        new_H = int(np.sqrt(new_HW))
        new_W = int(np.sqrt(new_HW))
        x = tf.reshape(x, [-1, new_H, new_W, C]) # [B, 56, 56, 64]
        # iteration1: soft split
        x = self.soft_split1(x) # [B, 28x28, 64x3x3]

        # iteration2: re-structurization/reconstruction
        x = self.performer2(x) # [B, 28x28, 64]
        B, new_HW, C = x.shape
        new_H = int(np.sqrt(new_HW))
        new_W = int(np.sqrt(new_HW))
        x = tf.reshape(x, [-1, new_H, new_W, C]) # [B, 28, 28, 64]
        # iteration2: soft split
        x = self.soft_split2(x) # [B, 14x14, 64x3x3]

        # final tokens
        x = self.project(x) # [B, 14x14, 768]

        return x


class T2T_ViT(tf.keras.Model):
    def __init__(self, image_size=224, tokens_type='performer', in_channels=3, num_classes=1000, hidden_size=768, depth=12,
                 num_heads=12, mlp_ratio=4., token_size=64, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.hidden_size = hidden_size  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(image_size=image_size, tokens_type=tokens_type, 
                                          in_channels=in_channels, embedding_size=hidden_size, token_size=token_size)
        num_patches = self.tokens_to_token.num_patches

        self.cls_tokens = self.add_weight('cls_tokens', shape=[1, 1, hidden_size],
                                    initializer=tf.keras.initializers.RandomNormal(),
                                    dtype=tf.float32)
        self.pos_embedding = self.add_weight('position_embedding', shape=[num_patches + 1, hidden_size], dtype=tf.float32, trainable=False)
        self.pos_embedding.assign(tf.squeeze(get_sinusoid_encoding(num_patches + 1, hidden_size)))

        self.transformer_encoders = TransformerEncoderBlock(hidden_size=hidden_size, num_layers=depth, num_heads=num_heads, 
                                                            intermediate_size=int(mlp_ratio * hidden_size), norm_first=True) 
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        # Classifier head
        self.classifier_head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

        # TODO: initial weights, not needed when only measuring inference latency
        # trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.tokens_to_token(x) # [B, N, H]
        broadcast_shape = tf.where([False, True, True], self.cls_tokens.shape, [tf.shape(x)[0], 0, 0], )
        cls_tokens = tf.broadcast_to(self.cls_tokens, broadcast_shape)
        x = tf.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embedding

        x = self.transformer_encoders(x)

        x = self.norm(x)
        return x[:, 0]

    def call(self, x):
        x = self.forward_features(x)
        x = self.classifier_head(x)
        return x


def get_t2t_vit_7():
    return T2T_ViT(hidden_size=256, depth=7, num_heads=4, mlp_ratio=2)

def get_t2t_vit_10():
    return T2T_ViT(hidden_size=256, depth=10, num_heads=4, mlp_ratio=2)

def get_t2t_vit_12():
    return T2T_ViT(hidden_size=256, depth=12, num_heads=4, mlp_ratio=2)

def get_t2t_vit_14():
    return T2T_ViT(hidden_size=384, depth=14, num_heads=6, mlp_ratio=3)