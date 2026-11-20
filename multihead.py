from tokenization import tokenizer
import torch
from head import Head
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention"""
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size, n_embd, block_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
