import torch
import torch.nn as nn
from torch.nn import functional as F
import config as cfg

# ======================================================
# Bonus 4E: KV-cache scaffold (intentionally guided)
# ======================================================
# Goal:
# - Speed up generation by caching past keys/values per layer.
#
# High-level approach:
# 1) Modify attention so it can accept `past_kv=(past_k, past_v)` and return `present_kv`.
# 2) Modify the LM forward to accept a list of per-layer caches and return updated caches.
# 3) In generate(), after the first step, feed ONLY the newest token (T_new=1) plus caches.

class MHAWithCache(nn.Module):
    """MHA that supports KV caching (skeleton).

    Shapes:
    - x: (B, T_new, C)
    - past_k: (B, n_head, T_past, head_dim)
    - past_v: (B, n_head, T_past, head_dim)

    Return:
    - out: (B, T_new, C)
    - present_kv: (k_total, v_total) with time length T_past + T_new

    Implementation outline:
    - Project q,k,v from x
    - Reshape to head format
    - If past exists: concatenate along time dim for k and v
    - Compute attention weights of q against k_total
    - If T_new > 1, enforce causality within the new chunk
    """
    def __init__(self):
        super().__init__()
        # TODO: store cfg.n_head, compute head_dim, build qkv projections and output proj, dropouts, mask buffer
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x, past_kv=None):
        # TODO: implement cache-aware attention and return (out, present_kv)
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(cfg.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        T_total = k.shape[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if T > 1:
            mask = self.mask[:T, :T_total]
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        present_kv = (k, v)
        return y, present_kv

class Block(nn.Module):
    """Transformer block that threads KV cache through attention."""
    def __init__(self):
        super().__init__()
        # TODO: norms, attention-with-cache, MLP
        self.attn = MHAWithCache()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x, past_kv=None):
        # TODO:
        # - call attention with past_kv
        # - add residuals
        # - return (x, present_kv)
        attn_out, present_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv

class KVCachedLanguageModel(nn.Module):
    """LM that accepts/returns a list of KV caches, one per layer."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: embeddings, ModuleList of blocks, final norm, lm head
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None, past_kv=None):
        """Forward with caches.

        past_kv:
          - None, OR
          - list length cfg.n_layer; each entry is (k, v) for that layer

        Return:
          logits, loss, presents
        """
        # TODO:
        # - embed tokens/positions
        # - for each layer, pass that layer's past cache (if any)
        # - collect `presents` list to return
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        presents = []
        for i, block in enumerate(self.blocks):
            past = past_kv[i] if past_kv else None
            x, present = block(x, past)
            presents.append(present)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss, presents

    @torch.no_grad()
    def generate_kvcache(self, idx, max_new_tokens: int):
        """Generate with caching.

        Guidance:
        - First step can run on a short prefix with cache=None
        - After that, feed only the last token (shape (B,1)) and the cache
        - Each step updates the cache
        """
        cache = None
        # TODO: implement sampling loop, updating cache each step
        for _ in range(max_new_tokens):
            if cache is None:
                logits, _, cache = self(idx, past_kv=None)
            else:
                logits, _, cache = self(idx[:, -1:], past_kv=cache)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
