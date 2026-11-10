"""Confirmed model and training configuration.

Architecture decisions (2026-11-10): 12 layers x 512 width with 8 heads, rotary
position embeddings, SwiGLU feed-forward, 1024-token context, tied embeddings, and
pre-norm RMSNorm. Batches and the token budget come from the measured throughput of
this GPU (see model/bench.py): micro-batch 8 at context 1024 with gradient
checkpointing uses ~2.3 GiB and sustains ~21,800 tokens/sec.
"""

from dataclasses import dataclass

from model.gpt import GPTConfig


ARCH = GPTConfig(
    vocab_size=16_000,
    max_seq_len=1024,
    n_layer=12,
    n_head=8,
    d_model=512,
    d_ff=1408,  # 8/3 * d_model rounded to a multiple of 128, the usual SwiGLU ratio
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
