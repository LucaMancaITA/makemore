
# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F


class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((vocab_size, vocab_size)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
