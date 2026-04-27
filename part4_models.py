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
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

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
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

class Block_MHA_RMS_SwiGLU(nn.Module):
    """Modern block: RMSNorm + MHA + SwiGLU.

    Structure:
    - x = x + Attn(RMSNorm(x))
    - x = x + MLP(RMSNorm(x))
    """
    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

class Block_GQA_RMS_SwiGLU(nn.Module):
    """Modern block: RMSNorm + GQA + SwiGLU."""
    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError

class MHA_ModernLanguageModel(nn.Module):
    """Full LM using modern MHA blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO:
        # - token embedding table
        # - position embedding table
        # - stack cfg.n_layer blocks
        # - final norm + lm_head
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # TODO
        raise NotImplementedError

class GQA_ModernLanguageModel(nn.Module):
    """Full LM using modern GQA blocks."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, idx, targets=None):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        # TODO
        raise NotImplementedError
