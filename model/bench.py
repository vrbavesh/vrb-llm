"""Verify the confirmed configuration on this GPU and report its throughput.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -u -m model.bench
"""

import argparse
from dataclasses import replace
import gc
import time

import torch

from model.config import ARCH
from model.gpt import GPT


def measure(cfg, batch, steps):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = GPT(cfg).cuda()
    params = model.num_params()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    x = torch.randint(1, cfg.vocab_size, (batch, cfg.max_seq_len), device="cuda")
    y = torch.randint(1, cfg.vocab_size, (batch, cfg.max_seq_len), device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(2):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = batch * cfg.max_seq_len * steps
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, x, y
    torch.cuda.empty_cache()
    gc.collect()
    return params, peak, tokens / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq", type=int, default=ARCH.max_seq_len)
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    cfg = replace(ARCH, max_seq_len=args.seq, checkpointing=not args.no_checkpointing)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} MiB), seq={cfg.max_seq_len}, "
          f"checkpointing={cfg.checkpointing}, mlp={cfg.mlp}, d_ff={cfg.d_ff}")
    print(f"{'batch':>6} {'tok/step':>9} {'peakMiB':>9} {'tok/s':>9}  result")

    for batch in args.batches:
        try:
            params, peak, tps = measure(cfg, batch, args.steps)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            print(f"{batch:>6} {'-':>9} {'-':>9} {'-':>9}  OOM")
            continue
        # Windows WDDM pages VRAM to host instead of raising, so an oversized peak
        # still completes at a tiny fraction of normal speed.
        result = "ok" if peak < vram * 0.96 else "SPILL (did not fit)"
        print(f"{batch:>6} {batch * cfg.max_seq_len:>9} {peak:>9.0f} {tps:>9.0f}  {result}")


if __name__ == "__main__":
    main()
