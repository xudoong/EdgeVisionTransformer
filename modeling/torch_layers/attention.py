import torch
import torch.nn as nn

class Attention(nn.Module):
    # reference huggingface ViTSelfAttention 
    def __init__(self, hidden_size, num_heads, head_size=None):
        if head_size is None:
            if hidden_size % num_heads != 0:
                raise ValueError(f'hidden_size {head_size} must be a multiple of num_heads {num_heads}.')
            self.head_size = hidden_size // num_heads
        else:
            self.head_size = head_size
        
        super().__init__()
        self.num_heads = num_heads
        self.scale = self.head_size ** -0.5

        self.to_query = nn.Linear(in_features=hidden_size, out_features=self.num_heads * self.head_size)
        self.to_key = nn.Linear(in_features=hidden_size, out_features=self.num_heads * self.head_size)
        self.to_value = nn.Linear(in_features=hidden_size, out_features=self.num_heads * self.head_size)
        self.to_out = nn.Linear(in_features=self.num_heads * self.head_size, out_features=hidden_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_query = self.to_query(x)

        key = self.transpose_for_scores(self.to_key(x))
        value = self.transpose_for_scores(self.to_value(x))
        query = self.transpose_for_scores(self.to_query(mixed_query))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * self.scale 

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()

        next_shape = context.size()[:-2] + (self.num_heads * self.head_size,)
        context = context.view(*next_shape)

        return context