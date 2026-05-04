import torch
import torch.nn as nn
from torch.nn import functional as F
import config as cfg

# ==========================
# Part 4: Extension skeleton
# ==========================
# You will implement:
# - Multi-head masked self-attention (MHA)
# - Grouped-query attention (GQA)
# - Modernization: RMSNorm + SwiGLU
# - LM variants that use those blocks

class RMSNorm(nn.Module):
    """RMSNorm: normalize by root-mean-square (no mean subtraction).

    What you should do:
    - Create a learnable scale parameter of shape (dim,)
    - In forward:
        * compute rms over the last dimension
        * divide x by rms (with eps for stability)
        * multiply by the learnable scale
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # TODO
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # TODO
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class SwiGLU(nn.Module):
    """SwiGLU MLP: gated feedforward.

    High-level idea:
    - Project x into TWO hidden vectors (same hidden size)
    - Apply SiLU to one branch and use it as a gate:
        gated = a * SiLU(b)
    - Project back to model dimension, then dropout

    What you should do:
    - Pick hidden size (commonly cfg.ff_mult * cfg.n_embd)
    - Implement gating + output projection (+ dropout)
    """
    def __init__(self, dim: int, mult: int):
        super().__init__()
        # TODO
        hidden_dim = mult * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # TODO
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class MultiHeadSelfAttention(nn.Module):
    """Masked Multi-Head Self-Attention (MHA).

    What you should build:
    - Ensure cfg.n_embd is divisible by cfg.n_head
    - Projections for queries/keys/values (separate or packed qkv)
    - Reshape into (B, n_head, T, head_dim)
    - Scaled dot-product attention with a causal mask
    - Concatenate heads back to (B, T, cfg.n_embd)
    - Output projection back to cfg.n_embd (+ dropout)
    """
    def __init__(self):
        super().__init__()
        # TODO
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                                     .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        # TODO
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class GroupedQuerySelfAttention(nn.Module):
    """Masked Grouped-Query Attention (GQA).

    You have:
    - cfg.n_head query heads
    - cfg.n_kv_head key/value heads
    Constraint:
    - cfg.n_head must be divisible by cfg.n_kv_head

    High-level plan:
    - Project queries to cfg.n_head heads
    - Project keys/values to cfg.n_kv_head heads
    - Repeat KV heads to align with query heads (share KV across groups)
    - Run masked attention
    """
    def __init__(self):
        super().__init__()
        # TODO
        assert cfg.n_head % cfg.n_kv_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                                     .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        # TODO
        B, T, C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # repeat k and v for each group
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class Block_MHA_RMS_SwiGLU(nn.Module):
    """Modern block: RMSNorm + MHA + SwiGLU.

    Structure:
    - x = x + Attn(RMSNorm(x))
    - x = x + MLP(RMSNorm(x))
    """
    def __init__(self):
        super().__init__()
        # TODO
        self.attn = MultiHeadSelfAttention()
        self.mlp = SwiGLU(cfg.n_embd, cfg.ff_mult)
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.mlp_norm = RMSNorm(cfg.n_embd)

    def forward(self, x):
        # TODO
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class Block_GQA_RMS_SwiGLU(nn.Module):
    """Modern block: RMSNorm + GQA + SwiGLU."""
    def __init__(self):
        super().__init__()
        # TODO
        self.attn = GroupedQuerySelfAttention()
        self.mlp = SwiGLU(cfg.n_embd, cfg.ff_mult)
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.mlp_norm = RMSNorm(cfg.n_embd)

    def forward(self, x):
        # TODO
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class MHA_ModernLanguageModel(nn.Module):
    """Full LM using modern MHA blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO:
        # - token embedding table
        # - position embedding table
        # - stack cfg.n_layer blocks
        # - final norm + lm_head
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block_MHA_RMS_SwiGLU() for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # TODO
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # TODO
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class GQA_ModernLanguageModel(nn.Module):
    """Full LM using modern GQA blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block_GQA_RMS_SwiGLU() for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # TODO
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # TODO
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
