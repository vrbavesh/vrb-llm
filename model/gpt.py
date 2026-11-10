"""Decoder-only transformer whose size is fully described by GPTConfig.

Depth, width, and feed-forward width are all configurable so the same code runs a
50M sanity check and a memory-ceiling benchmark on this machine.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16_000
    max_seq_len: int = 512
    n_layer: int = 12
    n_head: int = 8
    d_model: int = 512
    d_ff: int | None = None
    dropout: float = 0.0
    tie_embeddings: bool = True
    checkpointing: bool = False
    mlp: str = "swiglu"

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if (self.d_model // self.n_head) % 2:
            raise ValueError("head dimension must be even for rotary embeddings")
        if self.mlp not in ("swiglu", "gelu"):
            raise ValueError(f"unknown mlp type: {self.mlp}")
        if self.d_ff is None:
            if self.mlp == "swiglu":
                # 8/3 * d_model is the parameter-neutral SwiGLU width; round up to a
                # multiple of 128 so the projection stays a wide, efficient matmul.
                default = math.ceil(8 * self.d_model / 3 / 128) * 128
            else:
                default = 4 * self.d_model
            object.__setattr__(self, "d_ff", default)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(seq_len, head_dim, base=10_000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.arange(seq_len, device=device).float()[:, None] * inv[None, :]
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x, cos, sin):
    """x: (batch, heads, seq, head_dim); cos/sin: (1, 1, seq, head_dim / 2)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        batch, seq, d_model = x.shape
        head_dim = d_model // self.n_head
        q, k, v = self.qkv(x).view(batch, seq, 3, self.n_head, head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            apply_rope(q, cos, sin), apply_rope(k, cos, sin), v, is_causal=True
        )
        return self.proj(y.transpose(1, 2).reshape(batch, seq, d_model))


class SwiGLU(nn.Module):
    """Gate-and-up projection: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        if cfg.mlp == "swiglu":
            self.mlp = SwiGLU(cfg)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        seq = idx.shape[1]
        if seq > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len {self.cfg.max_seq_len}")
        cos, sin = self.cos[:, :, :seq], self.sin[:, :, :seq]
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.cfg.checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_seq_len :])
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k is not None:
                cutoff = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx
