import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, sub_layer):
        super().__init__()
        self.sub_layer = sub_layer

    def forward(self, x):
        return x + self.sub_layer(x)