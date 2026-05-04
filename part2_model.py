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
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits, loss).

        logits shape: (B, T, vocab_size)

        If targets is provided:
        - compute cross-entropy over all positions in the batch
        - return the scalar loss
        """
        logits = self.token_embedding_table(idx)  # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

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
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

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
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        v = self.value(x) # (B,T,head_size)
        
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B, T, head_size)
        return out

class FeedForward(nn.Module):
    """MLP used inside the Transformer block."""
    def __init__(self):
        super().__init__()
        # TODO: build a small MLP that expands then contracts back to n_embd
        # (expand -> nonlinearity -> project back -> dropout)
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlockSingleHead(nn.Module):
    """Transformer block using single-head attention."""
    def __init__(self):
        super().__init__()
        # TODO: create attention, feedforward, and norm modules
        head_size = n_embd
        self.sa = SingleHeadAttention(head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Apply pre-norm residual block.

        Structure:
        - residual attention: x = x + Attn(Norm(x))
        - residual MLP:       x = x + MLP(Norm(x))
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SingleHeadLanguageModel(nn.Module):
    """Decoder-only LM using single-head Transformer blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: token embedding table
        # TODO: positional embedding table up to block_size
        # TODO: stack of Transformer blocks
        # TODO: final norm + linear head to vocab logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlockSingleHead() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Compute logits (and optional loss).

        What you should do:
        - embed tokens and positions, add them
        - run through blocks
        - final norm and linear head to vocab
        - if targets provided, compute cross entropy over (B*T) positions
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Autoregressive generation with context windowing.

        What you should do:
        - at each step, condition on only the last `block_size` tokens
        - sample one new token and append
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
