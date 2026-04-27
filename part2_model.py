import torch
import torch.nn as nn
from torch.nn import functional as F
from config import n_embd, n_layer, block_size, dropout, device

class BigramBaseline(nn.Module):
    """True bigram model: next token distribution depends only on current token.

    What you should build:
    - A single embedding table that maps each token id -> logits over the vocab.
    - When idx is (B, T), the embedding lookup should produce (B, T, vocab_size).
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: create a learnable lookup that returns vocab-sized logits per token id
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """Return (logits, loss).

        logits shape: (B, T, vocab_size)

        If targets is provided:
        - compute cross-entropy over all positions in the batch
        - return the scalar loss
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Autoregressive sampling loop.

        What you should do each step:
        - run the model to get logits for the current sequence
        - take the logits for the last time-step
        - convert to probabilities (softmax)
        - sample the next token id
        - append it to the running sequence
        """
        raise NotImplementedError

class SingleHeadAttention(nn.Module):
    """One masked self-attention head.

    What you should build:
    - Linear projections for key, query, value
    - A causal mask stored as a buffer (lower-triangular) so it moves with device
    - Dropout on attention weights
    """
    def __init__(self, head_size: int):
        super().__init__()
        # TODO: define key/query/value projections
        # TODO: define and register a causal mask buffer
        # TODO: dropout module
        raise NotImplementedError

    def forward(self, x):
        """Compute masked self-attention.

        Steps you should implement:
        1) Project x into k, q, v
        2) Compute attention scores from q and k (scaled dot product)
        3) Apply causal mask so tokens cannot attend to the future
        4) Softmax over the last dimension to get weights
        5) Apply dropout to weights
        6) Weighted sum over v to get the output
        """
        raise NotImplementedError

class FeedForward(nn.Module):
    """MLP used inside the Transformer block."""
    def __init__(self):
        super().__init__()
        # TODO: build a small MLP that expands then contracts back to n_embd
        # (expand -> nonlinearity -> project back -> dropout)
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

class TransformerBlockSingleHead(nn.Module):
    """Transformer block using single-head attention."""
    def __init__(self):
        super().__init__()
        # TODO: create attention, feedforward, and norm modules
        raise NotImplementedError

    def forward(self, x):
        """Apply pre-norm residual block.

        Structure:
        - residual attention: x = x + Attn(Norm(x))
        - residual MLP:       x = x + MLP(Norm(x))
        """
        raise NotImplementedError

class SingleHeadLanguageModel(nn.Module):
    """Decoder-only LM using single-head Transformer blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: token embedding table
        # TODO: positional embedding table up to block_size
        # TODO: stack of Transformer blocks
        # TODO: final norm + linear head to vocab logits
        raise NotImplementedError

    def forward(self, idx, targets=None):
        """Compute logits (and optional loss).

        What you should do:
        - embed tokens and positions, add them
        - run through blocks
        - final norm and linear head to vocab
        - if targets provided, compute cross entropy over (B*T) positions
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Autoregressive generation with context windowing.

        What you should do:
        - at each step, condition on only the last `block_size` tokens
        - sample one new token and append
        """
        raise NotImplementedError
