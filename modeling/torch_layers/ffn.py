import torch
import torch.nn as nn
from torch.nn.modules import activation
from .activation import gelu


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = gelu(x)
        x = self.linear2(x)
        return x