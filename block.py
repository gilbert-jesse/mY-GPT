from multihead import MultiHeadAttention
from Feed_Forward import FeedForward
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head
        self.self_atten = MultiHeadAttention(n_embd, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.self_atten(x)
        x = self.ffwd(x)
        return x