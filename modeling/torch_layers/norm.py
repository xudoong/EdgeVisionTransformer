import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, input_shape, sub_layer, is_pre=False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_shape)
        self.sub_layer = sub_layer
        self.is_pre = is_pre

    def forward(self, x):
        if self.is_pre:
            return self.sub_layer(self.layer_norm(x))
        else:
            return self.layer_norm(self.sub_layer(x))