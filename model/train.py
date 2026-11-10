"""Train the confirmed model on the prepared token corpus.

The corpus is a flat uint16 array, so resuming is exact: the loop derives its
position from the optimizer step instead of maintaining dataloader state.
Checkpoints carry weights, Adam moments, and the step count, so a run can be
stopped at any point and continued later without a warm restart.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -u -m model.train --run-dir model/runs/main
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from model.config import ARCH, TRAIN
from model.gpt import GPT


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def lr_at(step, total, peak, warmup, min_ratio):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def batch_at(data, offset, micro, seq, device):
    chunk = np.array(data[offset : offset + micro * seq + 1], dtype=np.int64)
    x = torch.from_numpy(chunk[:-1]).reshape(micro, seq).to(device, non_blocking=True)
    y = torch.from_numpy(chunk[1:]).reshape(micro, seq).to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(model, data, device, micro, seq, token_budget):
    model.eval()
    total, seen = 0.0, 0
    span = micro * seq
    token_budget = min(token_budget, len(data))
    for offset in range(0, token_budget - span - 1, span):
        x, y = batch_at(data, offset, micro, seq, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total += loss.item() * y.numel()
        seen += y.numel()
    model.train()
    return total / max(seen, 1)


@torch.no_grad()
def sample(model, tokenizer, device, max_new_tokens=200):
    model.eval()
    idx = torch.tensor([tokenizer.encode("<bos>").ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature=0.8, top_k=50)
    model.train()
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(path, model, optimizer, step, val_loss, tokens_per_step, best_val=float("inf")):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val": best_val,
        "tokens_per_step": tokens_per_step,
        "tokens_seen": step * tokens_per_step,
        "arch": asdict(ARCH),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_history(run_dir, keep=3):
    history = sorted(run_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for path in history[:-keep]:
        path.unlink(missing_ok=True)


def log(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=TRAIN.total_steps)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else HERE.parent / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    samples_path = run_dir / "samples.txt"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    if not (args.data_dir / "train.bin").exists():
        raise SystemExit(f"missing corpus: {args.data_dir / 'train.bin'}; run `python -m model.data` first")

    torch.manual_seed(TRAIN.seed)
    device = "cuda"
    model = GPT(ARCH).cuda()
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.dim() >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": TRAIN.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=TRAIN.lr, betas=(TRAIN.beta1, TRAIN.beta2),
    )

    step, val_loss, best_val = 0, float("inf"), float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["tokens_per_step"] != TRAIN.tokens_per_step:
            raise SystemExit("checkpoint batch geometry does not match the current schedule")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        val_loss = checkpoint.get("val_loss", float("inf"))
        best_val = checkpoint.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at step {step}", flush=True)
    else:
        (run_dir / "config.json").write_text(
            json.dumps({"arch": asdict(ARCH), "train": asdict(TRAIN)}, indent=2))

    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")
    needed = args.max_steps * TRAIN.tokens_per_step + 1
    if len(train_data) < needed:
        raise SystemExit(f"corpus has {len(train_data):,} tokens but {needed:,} are needed")

    tokenizer = None
    if args.sample_every > 0:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(HERE.parent / "tokenizer" / "artifacts" / "tokenizer.json"))

    micro, seq = TRAIN.micro_batch, TRAIN.seq_len
    span = micro * seq
    print(f"training {model.num_params():,} params, steps {step} -> {args.max_steps}, "
          f"{TRAIN.tokens_per_step:,} tokens/step", flush=True)

    model.train()
    started = time.perf_counter()
    session_start_step = step
    try:
        while step < args.max_steps:
            lr = lr_at(step, args.max_steps, TRAIN.lr, TRAIN.warmup_steps, TRAIN.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            base = step * TRAIN.tokens_per_step
            loss_sum = 0.0
            for micro_step in range(TRAIN.grad_accum):
                x, y = batch_at(train_data, base + micro_step * span, micro, seq, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                (loss / TRAIN.grad_accum).backward()
                loss_sum += loss.item() / TRAIN.grad_accum

            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
            optimizer.step()
            step += 1

            elapsed = time.perf_counter() - started
            session_tokens = (step - session_start_step) * TRAIN.tokens_per_step
            record = {
                "step": step,
                "tokens": step * TRAIN.tokens_per_step,
                "loss": round(loss_sum, 5),
                "lr": round(lr, 7),
                "tok_per_s": round(session_tokens / elapsed),
                "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "elapsed_s": round(elapsed),
            }
            print(stdout_record(record), flush=True)
            log(metrics_path, record)

            if step % args.val_every == 0:
                val_loss = evaluate(model, val_data, device, micro, seq, args.val_tokens)
                print(f"  step {step}: val_loss {val_loss:.4f}", flush=True)
                log(metrics_path, {"step": step, "val_loss": round(val_loss, 5)})
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(best_path, model, optimizer, step, val_loss,
                                    TRAIN.tokens_per_step, best_val)

            if step % args.checkpoint_every == 0:
                save_checkpoint(last_path, model, optimizer, step, val_loss,
                                TRAIN.tokens_per_step, best_val)
                if step % 1000 == 0:
                    save_checkpoint(run_dir / f"step_{step}.pt", model, optimizer, step,
                                    val_loss, TRAIN.tokens_per_step, best_val)
                    prune_history(run_dir)

            if args.sample_every and step % args.sample_every == 0:
                text = sample(model, tokenizer, device)
                with open(samples_path, "a", encoding="utf-8") as handle:
                    handle.write(f"===== step {step} =====\n{text}\n\n")
                print(f"  step {step}: sample written", flush=True)

    except KeyboardInterrupt:
        print("interrupted; saving checkpoint", flush=True)
        save_checkpoint(last_path, model, optimizer, step, val_loss,
                        TRAIN.tokens_per_step, best_val)
        return 0
    except Exception:
        traceback.print_exc()
        try:
            save_checkpoint(last_path, model, optimizer, step, val_loss,
                            TRAIN.tokens_per_step, best_val)
            print(f"error at step {step}; checkpoint saved to {last_path}", flush=True)
        except Exception:
            print("checkpoint save also failed", flush=True)
        return 1

    save_checkpoint(last_path, model, optimizer, step, val_loss,
                    TRAIN.tokens_per_step, best_val)
    final = evaluate(model, val_data, device, micro, seq, args.val_tokens)
    print(f"done: step {step}, final val_loss {final:.4f}", flush=True)
    log(metrics_path, {"step": step, "final_val_loss": round(final, 5)})
    if tokenizer is not None:
        with open(samples_path, "a", encoding="utf-8") as handle:
            handle.write(f"===== final step {step} =====\n{sample(model, tokenizer, device, 300)}\n\n")
    return 0


def stdout_record(record):
    return (f"step {record['step']:>5}  loss {record['loss']:.4f}  lr {record['lr']:.2e}  "
            f"{record['tok_per_s']:>6} tok/s  {record['peak_mib']} MiB  {record['elapsed_s']}s")


if __name__ == "__main__":
    sys.exit(main())
