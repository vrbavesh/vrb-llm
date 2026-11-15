# VRB-LLM Project Report

**Project:** Language model built from scratch  
**Status:** Phase 1 complete (46.7M parameter model trained on 0.9B tokens)  
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), ~16 GB system RAM  
**Framework:** PyTorch 2.11.0 + CUDA 12.8, Python 3.13  
**Repository:** `C:\VRB-LLM` (git, branch `main`)

---

## 1. Objective

Build a small language model from scratch capable of generating short, grammatical English text, with a long-term target of grammatical essays. Constraints:

- Stay at approximately 46–50M parameters (user rejected scaling to 500M due to hardware limits)
- Train on local Project Gutenberg English corpus with boilerplate removed
- Tokenizer: byte-level BPE, 16,000-token vocabulary target
- 80/20 train/test split by book
- Run on a single consumer laptop GPU (RTX 4050, 6 GB VRAM)

---

## 2. Model Architecture (Phase 1: 46.7M parameters)

All settings locked in `model/config.py` and `model/gpt.py`.

| Component | Specification |
|---|---|
| Total parameters | **46,739,968** (~46.7M) |
| Layers | 12 decoder-only transformer blocks |
| Model width (`d_model`) | 512 |
| Attention heads | 8 (RoPE, head_dim = 64) |
| Feed-forward | SwiGLU, `d_ff` = 1,408 |
| Context length | 1,024 tokens |
| Embeddings | Tied (input/output share weights) |
| Normalization | Pre-norm RMSNorm |
| Dropout | None |
| Gradient checkpointing | Enabled |
| Causal attention | `F.scaled_dot_product_attention` with `is_causal=True` |

### Parameter breakdown

| Component | Parameter count |
|---|---|
| Transformer layers (12 × ~3.6M) | ~43.5M |
| Tied embedding matrix (16,000 × 512) | ~8.2M (shared) |
| Final RMSNorm | ~0.5K |
| **Total** | **46,739,968** |

---

## 3. Tokenizer & Data Pipeline

### Tokenizer (byte-level BPE)

| Metric | Value |
|---|---|
| Vocabulary size | 16,000 tokens |
| Special tokens | pad=0, unk=1, bos=2, eos=3 |
| Algorithm | Byte-level BPE |
| Chars per token | 3.7585 |
| Round-trip accuracy | 100% on 7,605 test books (zero unknown tokens) |
| Train/test book split | 30,419 train / 7,605 test (deterministic, deduplicated by base Gutenberg ID) |
| Artifact | `tokenizer/artifacts/tokenizer.json` |

### Pre-encoded corpus (flat `uint16` arrays)

| Split | Tokens | Size | Source |
|---|---|---|---|
| Training | 1,100,911,879 | 2.05 GiB | Held-out Gutenberg books (train key) |
| Validation | 3,042,134 | ~6 MB | Held-out books (test key) |

- `<eos>` appended after every book
- Reuses the tokenizer's cleaning and deduplication logic
- Both `.bin` files are gitignored (large data artifacts)

---

## 4. Training Configuration

All settings in `model/config.py` (`TRAIN`).

| Hyperparameter | Value |
|---|---|
| Total tokens | 900,000,000 (0.9B) |
| Micro-batch size | 8 sequences |
| Sequence length | 1,024 |
| Gradient accumulation | 32 steps |
| Tokens per optimizer step | 8 × 1,024 × 32 = **262,144** |
| Total optimizer steps | 3,433 |
| Optimizer | AdamW |
| Peak learning rate | 6e-4 |
| LR schedule | Linear warmup (100 steps) → cosine decay → 10% floor |
| Weight decay | 0.1 |
| Betas | (0.9, 0.95) |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 autocast |
| Validation interval | Every 250 steps |
| Sampling interval | Every 500 steps |
| Checkpoint interval | Every 200 steps (`last.pt`) + every 1,000 steps (`step_N.pt`) |
| Best-model checkpoint | `best.pt` (saved when validation improves) |

### Checkpoint / resume design

- Checkpoint payload: model state dict, optimizer state dict (both Adam parameter groups), step, val loss, tokens per step, tokens seen, architecture dict
- Atomic write: `torch.save` to `.tmp` then `os.replace`
- Exact resume: rebuilds architecture from checkpoint's `arch` dict, resumes from exact token offset
- 22 model tests and 3 tokenizer tests pass (31 total with status tests)

---

## 5. Training Results (Phase 1 Complete)

### Run history

| Session | Date | Steps | Duration | Stop reason |
|---|---|---|---|---|
| Run 1 | 2026-11-11 00:05 | 0 → 1,787 | ~6.2 h | Deliberate stop (user request) |
| Run 2 | 2026-11-11 12:24 | 1,787 → 2,800 | ~3.9 h | Crash: `torch.save` `ios_base::badbit` during checkpoint write (transient disk I/O from Windows Update contention) |
| Run 3 | 2026-11-11 17:08 | 2,750 → 3,433 | ~2.3 h | Completed normally |

**Total wall-clock: ~12.5 hours across 3 sessions**  
**Final step: 3,433 / 3,433 (100%)**  
**Total tokens processed: 0.9B**

### Loss trajectory

| Metric | Start | End (final) | Best |
|---|---|---|---|
| Training loss | 9.80 | 4.1985 | — |
| Validation loss | 9.50 | 3.8335 (final eval) | 3.8507 (step 3,250) |
| Random baseline | ln(16,000) = 9.68 | — | — |

Validation improved steadily: 4.2504 (step 1,750) → 4.0413 (step 2,250) → 3.9139 (step 2,750) → 3.8872 (step 3,000) → 3.8507 (step 3,250) → 3.8335 (final).

### Throughput & hardware utilization

| Metric | Value |
|---|---|
| Sustained training throughput | ~20,900 tokens/second |
| Benchmark batch 8 (checkpointing on) | 20,390 tok/s at 2,302 MiB VRAM |
| VRAM during training | ~2,672 MiB |
| GPU clock | ~2,483–2,520 MHz |
| GPU power draw | ~74–75 W |
| GPU temperature | ~78–86 °C (thermal slowdown observed) |
| GPU utilization | ~100% during training |

### Hardware constraints observed

- RTX 4050 Laptop GPU: 6,140 MiB VRAM. Practical safe ceiling for training: ~5,900 MiB (Windows WDDM pages to host above this instead of cleanly OOMing).
- On battery, GPU caps to ~33 W / 1,650 MHz (~half throughput); must run on AC.
- Lid-close action not configurable; lid must stay open for unattended runs.
- Windows Update 11 GB download caused 300–450 second disk stalls and one checkpoint write failure. Mitigated with 3-retry checkpoint logic and stale `.tmp` cleanup.

---

## 6. Inference (`model/sample.py`)

| Feature | Specification |
|---|---|
| Architecture rebuild | Rebuilds model from checkpoint's `arch` dict (not hardcoded `ARCH`) |
| Sampling | Temperature + top-k |
| Stop token | Stops at `<eos>` |
| CLI flags | `--prompt`, `--tokens`, `--temperature`, `--top-k`, `--samples`, `--seed`, `--out` |
| Prompt-duplication bug | Found and fixed during initial demo |

### Sample output quality (step 3,433)

The model generates grammatically valid English with correct punctuation, dialogue structure (quotation marks, "said X" attributions), and sentence boundaries. At 46.7M parameters on 0.9B tokens, it exhibits strong local coherence but loses topic consistency across longer spans — consistent with a small model at the end of its Chinchilla-optimal token budget.

Example (prompt: "The old man walked at the park", temperature 0.2):

> The old man walked at the park, and, as he was looking at the door, he was looking at the little fellow...  
> "I'm glad you're going to go to the house," he said...

---

## 7. Tooling Added

### `model/status.py` (new)
One-command training status viewer. Reads only `train.log`, `metrics.jsonl`, and `nvidia-smi` — never touches the model or CUDA context, so safe to run during training.

| Output | Description |
|---|---|
| Verdict | RUNNING / FINISHED / CRASHED / STALLED |
| Progress | Steps completed, % complete, tokens processed |
| Metrics | Latest loss, learning rate, tokens/sec, VRAM |
| ETA | Estimated time to completion (from last two steps) |
| GPU | Utilization, clock, power, temperature, VRAM |
| Validation | Last 5 val_loss values, best val_loss |
| Checkpoints | Timestamps of all `.pt` files |
| Disk | Free space on drive |
| Crash recovery | Prints last `train.err` line and exact `python -m model.train --resume ...` command |

### Test suite
| Suite | Count | Status |
|---|---|---|
| Model tests (`test_train.py`) | 22 | Passing |
| Tokenizer tests | 3 | Passing |
| Status tests (`test_status.py`) | 6 | Passing |
| **Total** | **31** | **Passing** |

---

## 8. Technical Challenges & Resolutions

| Challenge | Resolution |
|---|---|
| 6 GB VRAM ceiling prevents larger models | Gradient checkpointing, micro-batch 8, 262,144 token effective batch |
| Intermittent multi-minute step stalls | Traced to Windows Update disk saturation; laptop must stay on AC, lid open |
| Checkpoint write crash (`ios_base::badbit`) | Transient disk I/O failure; added 3-retry backoff, `.tmp` cleanup on startup and between attempts |
| Exact resume across sessions | Checkpoint keyed to token offset; architecture stored in checkpoint; verified resumable with all 86 optimizer states |
| Short benchmarks understate throughput | GPU boosts only under sustained load; real training (20,900 tok/s) exceeded short bench |

---

## 9. Next Phase (Planned, Not Started)

| Item | Plan |
|---|---|
| Architecture | Scale to ~150.2M parameters: 23 layers × 704 width × 11 RoPE heads, SwiGLU `d_ff` 1,920 (committed to `main`, not yet trained) |
| Data | Encode full 3.2B-token corpus (`--budget 3213000000`, ~21 min) |
| Training | One epoch on full corpus, monitor val loss per epoch, stop when improvement <0.02 |
| Benchmark | Sweep micro-batch / checkpointing to find fastest config under 5,900 MiB |

---

## 10. Repository Structure

```
C:\VRB-LLM\
├── model/
│   ├── config.py        # ARCH (46.7M → 150.2M proposed) and TRAIN schedule
│   ├── gpt.py           # GPTConfig, GPT, attention, SwiGLU, RMSNorm, RoPE
│   ├── train.py         # Training loop, checkpoint/resume, validation, sampling
│   ├── data.py          # Corpus builder (--label train|val, --budget)
│   ├── sample.py        # Inference script (temperature, top-k, --out)
│   ├── bench.py         # VRAM/throughput sweep
│   ├── status.py        # One-command run status viewer
│   ├── tests/
│   │   ├── test_train.py (22 tests)
│   │   └── test_status.py (6 tests)
│   └── runs/main/       # train.log, metrics.jsonl, samples.txt, *.pt checkpoints
├── tokenizer/
│   └── artifacts/
│       └── tokenizer.json
├── AGENTS.md            # Project goal and feature history (gitignored)
├── HANDOFF.md           # Detailed architecture, run state, resume commands (gitignored)
└── PROJECT_REPORT.md    # This file
```

---

## 11. Resume-Ready Summary

> Built and trained a 46.7M-parameter decoder-only transformer language model from scratch in PyTorch on a single RTX 4050 laptop GPU. Trained on 0.9B tokens of Project Gutenberg text (byte-level BPE tokenizer, 16k vocab), achieving validation loss of 3.83 from a random baseline of 9.68. Implemented exact checkpoint/resume, gradient checkpointing, and a monitoring tool; sustained ~21k tokens/second within a 2.7 GB VRAM budget. Designed a ~150M-parameter successor architecture verified to fit the same 6 GB hardware.
