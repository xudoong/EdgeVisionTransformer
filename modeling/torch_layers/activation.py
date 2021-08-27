import math
import torch

def gelu(x):
    cdf = 0.5 * (1.0 + torch.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf