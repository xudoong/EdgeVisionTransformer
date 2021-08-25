from models.vit import ViT
import tensorflow as tf


if __name__ == '__main__':
    vit_config = {
        "image_size":224,
        "patch_size":16,
        "num_classes":1000,
        "dim":768,
        "depth":12,
        "heads":12,
        "mlp_dim":3072
    }

    vit = ViT(**vit_config)
    vit = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3, vit_config["image_size"], vit_config["image_size"]), batch_size=1),
        vit,
    ])

    vit.save(f'/data/v-xudongwang/models/tf_model/vit_test_patch16_224.tf')