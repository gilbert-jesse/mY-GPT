from tokenization import tokenizer
import torch
from head import Head
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention"""
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size, n_embd, block_size) for _ in range(num_heads))

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
