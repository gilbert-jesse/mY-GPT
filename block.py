from multihead import MultiHeadAttention
from Feed_Forward import FeedForward
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size  = n_embd // n_head
        self.self_atten = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size)
        self.feedfwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.self_atten(x)
        x = x + self.feedfwd(x)
        return x