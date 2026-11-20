import torch
import torch.nn as nn
from torch.nn import functional as F
from block import Block
from multihead import MultiHeadAttention
from Feed_Forward import FeedForward

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available else "cpu"

class BiagramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),
            Block(n_embd, n_head=4, block_size=block_size),

        )

        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size=block_size

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) #(T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, Vocab_size)

        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the context size because the position embeddings have size context, embeddings
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only in the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # B, C
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the runnning sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


