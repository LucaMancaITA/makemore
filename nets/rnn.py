
# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F


class RNNCell(nn.Module):
    """Recurrent Neural Network cell.

    It takes input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state h_{t} at
    the current timestep.
    """
    def __init__(self, n_embd1, n_embd2):
        super().__init__()
        self.xh_to_h = nn.Linear(n_embd1 + n_embd2, n_embd2)

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = F.tanh(self.xh_to_h(xh))
        return ht

class GRUCell(nn.Module):
    """Gated Recurrent Unit cell."""
    def __init__(self, n_embd1, n_embd2):
        super().__init__()
        # Input gate
        self.xh_to_z = nn.Linear(n_embd1 + n_embd2, n_embd2)
        # Forget gate
        self.xh_to_r = nn.Linear(n_embd1 + n_embd2, n_embd2)
        # Output gate
        self.xh_to_hbar = nn.Linear(n_embd1 + n_embd2, n_embd2)

    def forward(self, xt, hprev):
        # Reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # Calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = F.tanh(self.xh_to_hbar(xhr))
        # Calculate the switch gate that determines if each channel should be
        # updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # Blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class RNN(nn.Module):

    def __init__(self, block_size, vocab_size, n_embd1, n_embd2,
                 cell_type="rnn"):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.start = nn.Parameter(torch.zeros(1, n_embd2)) # the starting hidden state
        self.wte = nn.Embedding(vocab_size, n_embd1) # token embeddings table
        if cell_type == 'rnn':
            self.cell = RNNCell(n_embd1, n_embd2)
        elif cell_type == "gru":
            self.cell = GRUCell(n_embd1, n_embd2)
        self.lm_head = nn.Linear(n_embd2, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()

        # Embed all the integers up front and all at once for efficiency
        emb = self.wte(idx) # (b, t, n_embd)

        # Sequentially iterate over the inputs and update the RNN state
        hprev = self.start.expand((b, -1)) # expand out the batch dimension
        hiddens = []
        for i in range(t):
            xt = emb[:, i, :] # (b, n_embd)
            ht = self.cell(xt, hprev) # (b, n_embd2)
            hprev = ht
            hiddens.append(ht)

        # Decode the outputs
        hidden = torch.stack(hiddens, 1) # (b, t, n_embd2)
        logits = self.lm_head(hidden)

        # Loss function
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1)

        return logits, loss
