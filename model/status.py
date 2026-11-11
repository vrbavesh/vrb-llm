"""One-command view of the training run.

Reads the run directory and asks nvidia-smi; it never touches the model, the
optimizer, or CUDA context, so it is safe to run while training is in progress.

    python -m model.status
    python -m model.status --watch 60
    python -m model.status --run-dir model/runs/main
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:  # allow `python model/status.py`
    sys.path.insert(0, str(ROOT))

DEFAULT_RUN = ROOT / "model" / "runs" / "main"

# A step line looks like:
#   step  3048  loss 4.2674  lr 7.77e-05   20892 tok/s  2672 MiB  3739s
STEP_RE = re.compile(
    r"^step\s+(\d+)\s+loss\s+(-?[\d.]+)\s+lr\s+([\d.eE+-]+)\s+"
    r"([\d,]+)\s+tok/s\s+(\d+)\s+MiB\s+(\d+)s"
)

# Generous enough to cover a validation pass plus a 535 MB checkpoint write.
STALE_SECONDS = 150


def read_steps(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = STEP_RE.match(line.strip())
            if match:
                rows.append(
                    {
                        "step": int(match.group(1)),
                        "loss": float(match.group(2)),
                        "lr": float(match.group(3)),
                        "tok_s": float(match.group(4).replace(",", "")),
                        "mib": int(match.group(5)),
                        "elapsed": int(match.group(6)),
                    }
                )
    return rows


def read_vals(metrics_path: Path, limit: int = 5) -> list[tuple[int, float]]:
    vals: list[tuple[int, float]] = []
    if not metrics_path.exists():
        return vals
    with metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" in record:
                vals.append((int(record["step"]), float(record["val_loss"])))
    return vals[-limit:]


def gpu_snapshot() -> dict | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu,"
                "memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    fields = ["util", "clock", "power", "temp", "vram", "vram_total"]
    parts = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(fields, parts))


def checkpoint_files(run_dir: Path) -> list[tuple[str, float, datetime]]:
    files = []
    for path in run_dir.glob("*.pt"):
        stat = path.stat()
        files.append((path.name, stat.st_size / 2**20, datetime.fromtimestamp(stat.st_mtime)))
    return sorted(files, key=lambda item: item[2])


def format_duration(seconds: float) -> str:
    if seconds < 0:
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def train_config() -> tuple[int | None, int | None]:
    """Return (total_steps, tokens_per_step), or (None, None) if unavailable."""
    try:
        from model.config import TRAIN

        return TRAIN.total_steps, TRAIN.tokens_per_step
    except Exception:
        return None, None


def sample_steps(samples_path: Path) -> list[int]:
    if not samples_path.exists():
        return []
    found = re.findall(r"^===== step (\d+) =====", samples_path.read_text(
        encoding="utf-8", errors="replace"), flags=re.MULTILINE)
    return [int(value) for value in found]


def report(run_dir: Path, val_limit: int = 5) -> str:
    lines: list[str] = []
    log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.jsonl"
    err_path = run_dir / "train.err"
    samples_path = run_dir / "samples.txt"

    rows = read_steps(log_path)
    vals = read_vals(metrics_path, val_limit)
    err_text = ""
    if err_path.exists():
        err_text = err_path.read_text(encoding="utf-8", errors="replace").strip()
    target, tokens_per_step = train_config()

    lines.append(f"VRB-LLM training status   {datetime.now():%H:%M:%S}")
    lines.append(f"run  {run_dir}")

    if not rows:
        lines.append("")
        lines.append("NO STEP LINES in train.log -- the run has not started or the log is gone.")
        if err_text:
            lines.append(f"train.err: {err_text.splitlines()[-1]}")
        return "\n".join(lines)

    last = rows[-1]
    step = last["step"]
    age = time.time() - log_path.stat().st_mtime

    if target is None:
        finished = False
    else:
        finished = step >= target

    if finished:
        verdict = "FINISHED"
    elif age < STALE_SECONDS:
        verdict = "RUNNING"
    elif err_text:
        verdict = "CRASHED"
    else:
        verdict = "STALLED"

    lines.append("")
    lines.append(f"verdict  {verdict}   (last log line {format_duration(age)} ago)")

    if target and tokens_per_step:
        pct = 100.0 * step / target
        lines.append(f"progress {step:,} / {target:,} steps  ({pct:.1f}%)  "
                     f"{step * tokens_per_step / 1e9:.3f}B of "
                     f"{target * tokens_per_step / 1e9:.3f}B tokens")
    else:
        lines.append(f"progress {step:,} steps")

    lines.append(
        f"loss     {last['loss']:.4f}   lr {last['lr']:.2e}   "
        f"{last['tok_s']:,.0f} tok/s   {last['mib']} MiB"
    )

    # Token rate from the two most recent lines, so a session restart does not skew it.
    if target and not finished and len(rows) >= 2:
        step_delta = rows[-1]["step"] - rows[-2]["step"]
        time_delta = rows[-1]["elapsed"] - rows[-2]["elapsed"]
        if step_delta > 0 and time_delta > 0:
            seconds_per_step = time_delta / step_delta
            remaining = (target - step) * seconds_per_step
            finish_at = datetime.now() + timedelta(seconds=remaining)
            lines.append(f"eta      {format_duration(remaining)}  "
                         f"({seconds_per_step:.1f} s/step, done ~{finish_at:%H:%M})")

    snapshot = gpu_snapshot()
    if snapshot:
        lines.append(
            f"gpu      {snapshot['util']}%  {snapshot['clock']} MHz  "
            f"{snapshot['power']} W  {snapshot['temp']} C  "
            f"vram {snapshot['vram']}/{snapshot['vram_total']} MiB"
        )
    else:
        lines.append("gpu      unavailable")

    if vals:
        trend = "  ".join(f"{s:,} {v:.4f}" for s, v in vals)
        best = min(v for _, v in vals)
        lines.append(f"val      {trend}   (best {best:.4f})")
    else:
        lines.append("val      no validation recorded yet")

    files = checkpoint_files(run_dir)
    if files:
        shown = ", ".join(f"{name} {when:%H:%M:%S}" for name, _, when in files[-4:])
        lines.append(f"ckpt     {shown}")
    else:
        lines.append("ckpt     none")

    stray = list(run_dir.glob("*.tmp"))
    if stray:
        lines.append(f"WARNING  partial write left behind: {', '.join(p.name for p in stray)}")

    samples = sample_steps(samples_path)
    if samples:
        lines.append(f"samples  last at step {samples[-1]:,}")

    free = shutil.disk_usage(ROOT).free / 2**30
    lines.append(f"disk     {free:.2f} GB free")

    if verdict == "CRASHED":
        resume_path = run_dir / "best.pt"
        try:
            shown_dir = run_dir.relative_to(ROOT).as_posix()
            shown_resume = resume_path.relative_to(ROOT).as_posix()
        except ValueError:
            shown_dir, shown_resume = str(run_dir), str(resume_path)
        lines.append("")
        lines.append("last line of train.err:")
        lines.append("  " + err_text.splitlines()[-1])
        lines.append("resume with:")
        lines.append(f"  python -m model.train --run-dir {shown_dir} --resume {shown_resume}")
    elif verdict == "STALLED":
        lines.append("")
        lines.append("The log has not advanced. Check that the process is still alive;")
        lines.append("Python may be paging or blocked on disk.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show training run status.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN,
                        help="run directory containing train.log and metrics.jsonl")
    parser.add_argument("--watch", type=int, metavar="SECONDS", default=0,
                        help="refresh every N seconds instead of printing once")
    parser.add_argument("--lines", type=int, default=5,
                        help="how many validation points to show")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir

    if not args.watch:
        print(report(run_dir, args.lines))
        return

    while True:
        text = report(run_dir, args.lines)
        if args.watch:
            print("\n" * 2, end="")
        print(text, flush=True)
        verdict = [ln for ln in text.splitlines() if ln.startswith("verdict")]
        if verdict and ("FINISHED" in verdict[0] or "CRASHED" in verdict[0]):
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
