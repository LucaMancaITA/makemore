
# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    """Multi-layer perceptron.

    Embedding layer -> FC -> tanh -> FC.
    """
    def __init__(self, block_size, vocab_size, n_embd1, n_embd2):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd1 = n_embd1
        self.n_embd2 = n_embd2
        self.wte = nn.Embedding(self.vocab_size + 1, self.n_embd1)
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * self.n_embd1, self.n_embd2),
            nn.Tanh(),
            nn.Linear(self.n_embd2, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # Gather the word embeddings
        embs = []
        # Loop through the number of past characters to use to make the preds
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd1)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)

        # Concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # (b, t, n_embd1 * block_size)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
