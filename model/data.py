"""Build packed uint16 token shards for training and validation.

Streams the Project Gutenberg parquet shards, applies the same cleaning and
deduplication used to train the tokenizer, and writes a flat array of token IDs
with <eos> after every book. A flat array is what makes resuming training exact:
the loop's position is just a token offset.

Run from the project root:
    .\\.venv\\Scripts\\python.exe -u -m model.data --label train --budget 1100000000
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from tokenizer.train import clean_book


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "project_gutenberg" / "data"
ARTIFACTS = ROOT / "tokenizer" / "artifacts"
OUTPUT = Path(__file__).resolve().parent / "data"
EOS = 3
BATCH_BOOKS = 16


def flat_ids(tokenizer, texts):
    """Encode books and append <eos> to each, preserving order."""
    ids = []
    for encoding in tokenizer.encode_batch(texts):
        ids.extend(encoding.ids)
        ids.append(EOS)
    return ids


def build(label, budget, tokenizer, files, selected, out_path):
    seen = set()
    written = books = chars = 0
    pending, pending_chars = [], []
    start = time.perf_counter()

    def flush(handle):
        nonlocal written, books, chars, pending, pending_chars
        if not pending:
            return
        ids = flat_ids(tokenizer, pending)
        np.array(ids, dtype=np.uint16).tofile(handle)
        written += len(ids)
        books += len(pending)
        chars += sum(pending_chars)
        pending, pending_chars = [], []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        for name in files:
            if written >= budget:
                break
            for batch in pq.ParquetFile(DATA / name).iter_batches(batch_size=8, columns=["id", "text"]):
                for row in batch.to_pylist():
                    book_id = row["id"].partition("-")[0]
                    if book_id not in selected or book_id in seen:
                        continue
                    seen.add(book_id)
                    text, _, _ = clean_book(row["text"])
                    pending.append(text)
                    pending_chars.append(len(text))
                    if len(pending) >= BATCH_BOOKS:
                        flush(handle)
                        if books % 320 == 0:
                            rate = written / (time.perf_counter() - start)
                            eta = (budget - written) / rate / 60 if rate else 0
                            print(f"{label}: {written:,} tokens, {books:,} books, "
                                  f"{rate:,.0f} tok/s, eta {eta:.1f} min", flush=True)
                        if written >= budget:
                            break
                if written >= budget:
                    break
        flush(handle)

    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "books": books,
        "tokens": written,
        "characters": chars,
        "chars_per_token": chars / written if written else 0,
        "budget": budget,
        "bytes": out_path.stat().st_size,
        "seconds": round(elapsed, 1),
        "tokenizer_sha256": hashlib.sha256((ARTIFACTS / "tokenizer.json").read_bytes()).hexdigest(),
        "file": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["train", "val"], default="train")
    parser.add_argument("--budget", type=int, default=1_100_000_000)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split = json.loads((ARTIFACTS / "split.json").read_text())
    # split.json names the held-out books "test"; the corpus builder calls them "val".
    split_key = {"train": "train", "val": "test"}[args.label]
    selected = set(split[split_key])
    out_path = args.out_dir / f"{args.label}.bin"
    meta_path = args.out_dir / f"{args.label}.meta.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    tokenizer = Tokenizer.from_file(str(ARTIFACTS / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    print(f"{args.label}: {len(selected):,} books available, budget {args.budget:,} tokens", flush=True)
    meta = build(args.label, args.budget, tokenizer, split["files"], selected, out_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"{args.label}: wrote {meta['tokens']:,} tokens ({meta['bytes'] / 2**20:.2f} MiB) "
          f"in {meta['seconds'] / 60:.1f} min, {meta['chars_per_token']:.4f} chars/token", flush=True)
    print(f"{args.label}: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
