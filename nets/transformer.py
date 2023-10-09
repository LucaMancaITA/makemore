
# Imports
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo
    (identical to OpenAI GPT).
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = self.n_embd // self.n_head
        self.block_size = block_size
        # Query, key and value
        self.c_attn = nn.Linear(n_embd, n_embd*3)
        # Output projection
        self.fc_out = nn.Linear(n_embd, n_embd)
        # Causal mask to ensure that attention is only applied to the left in
        # the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size() # (batch_size, sequence_length, n_embd)

        # Query, key and value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention
        # Self-attend: (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, T)
        attn = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = attn @ v    # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.fc_out(y)

        return y


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc = nn.Linear(n_embd, 4*n_embd),
            c_proj = nn.Linear(4*n_embd, n_embd),
            act = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x):
        #x = x + self.attn(self.ln1(x))
        #x = x + self.mlpf(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlpf(x))

        return x


class Transformer(nn.Module):
    """Transformer language model."""

    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.n_embd),
            wpe = nn.Embedding(self.block_size, self.n_embd),
            h = nn.ModuleList([
                Block(self.n_embd, self.n_head, self.block_size) \
                    for _ in range(self.n_layer)]
            ),
            ln_f = nn.LayerNorm(self.n_embd)
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, \
            block size is only {self.block_size}"

        # Positional encoding
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0) # shape (1, t)
        pos_emb = self.transformer.wpe(pos) # (1, t, n_embd)

        # Input embedding
        inp_emb = self.transformer.wte(idx) # (b, t, n_embd)

        # Concatenate
        x = pos_emb + inp_emb

        # Multi-head self-attentions
        for block in self.transformer.h:
            x = block(x)

        # Multi-layer perceptron
        logits = self.transformer.ln_f(x)

        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1)

        return logits, loss
