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
        raise NotImplementedError

    def forward(self, x, past_kv=None):
        # TODO: implement cache-aware attention and return (out, present_kv)
        raise NotImplementedError

class Block(nn.Module):
    """Transformer block that threads KV cache through attention."""
    def __init__(self):
        super().__init__()
        # TODO: norms, attention-with-cache, MLP
        raise NotImplementedError

    def forward(self, x, past_kv=None):
        # TODO:
        # - call attention with past_kv
        # - add residuals
        # - return (x, present_kv)
        raise NotImplementedError

class KVCachedLanguageModel(nn.Module):
    """LM that accepts/returns a list of KV caches, one per layer."""
    def __init__(self, vocab_size: int):
        super().__init__()
        # TODO: embeddings, ModuleList of blocks, final norm, lm head
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
