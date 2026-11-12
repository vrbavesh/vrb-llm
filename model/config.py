"""Confirmed model and training configuration.

Architecture decisions (2026-11-11): 23 layers x 704 width with 11 rotary heads
(head_dim 64), SwiGLU feed-forward at d_ff 1920, 1024-token context, tied
embeddings, pre-norm RMSNorm, no dropout, gradient checkpointing on — ~150.2M
parameters. Batches and the token budget come from the measured throughput of
this GPU (see model/bench.py).
"""

from dataclasses import dataclass

from model.gpt import GPTConfig


ARCH = GPTConfig(
    vocab_size=16_000,
    max_seq_len=1024,
    n_layer=23,
    n_head=11,
    d_model=704,
    d_ff=1920,  # ~2.73 * d_model, rounded to a multiple of 128, SwiGLU ratio
    dropout=0.0,
    tie_embeddings=True,
    checkpointing=True,
    mlp="swiglu",
)


@dataclass(frozen=True)
class TrainConfig:
    micro_batch: int = 8
    seq_len: int = 1024
    grad_accum: int = 32
    total_tokens: int = 900_000_000  # Chinchilla-optimal for ~46M parameters
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 1337

    @property
    def tokens_per_step(self):
        return self.micro_batch * self.seq_len * self.grad_accum

    @property
    def total_steps(self):
        return self.total_tokens // self.tokens_per_step


TRAIN = TrainConfig()
