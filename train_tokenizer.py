"""Train the configured tokenizer on the English Gutenberg training books."""

import hashlib
import json
from pathlib import Path
import re
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from tokenizer import SPECIAL_TOKENS, VOCAB_SIZE, tokenizer, trainer


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "project_gutenberg" / "data"
OUTPUT = ROOT / "artifacts" / "tokenizer"
MARKER = r"(?im)^[ \t]*\*{3}[ \t]*%s OF (?:THE|THIS) PROJECT GUTENBERG[^\r\n]*(?:\r?\n|$)"
START = re.compile(MARKER % "START")
END = re.compile(MARKER % "END")


def clean_book(text):
    # shortcut: unmarked boilerplate is retained; extend patterns after reviewing unmatched books.
    start = START.search(text)
    body_start = start.end() if start else 0
    end = END.search(text, body_start)
    return text[body_start:end.start() if end else len(text)], bool(start), bool(end)


def split_books(ids):
    if any(not isinstance(book_id, str) or not book_id for book_id in ids):
        raise ValueError("Every book must have a nonempty string ID")
    # Gutenberg suffixes identify alternate encodings of the same book.
    book_ids = {book_id.partition("-")[0] for book_id in ids}
    if any(not book_id for book_id in book_ids):
        raise ValueError("Expected nonempty Gutenberg book IDs before any encoding suffix")
    ordered = sorted(book_ids, key=lambda book_id: hashlib.sha256(("42:" + book_id).encode()).digest())
    boundary = len(ordered) * 80 // 100
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("Need enough books for nonempty training and test splits")
    return ordered[:boundary], ordered[boundary:]


def books(files, selected, stats):
    seen = set()
    for path in files:
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=["id", "text"], use_threads=False):
            for row in batch.to_pylist():
                book_id = row["id"].partition("-")[0]
                if book_id not in selected:
                    continue
                if book_id in seen:
                    stats["duplicate_rows_skipped"] += 1
                    continue
                seen.add(book_id)
                if not isinstance(row["text"], str):
                    raise ValueError(f"Non-text book: {row['id']}")
                text, header, footer = clean_book(row["text"])
                stats["books"] += 1
                stats["headers_removed"] += header
                stats["footers_removed"] += footer
                if not text.strip():
                    stats["empty_books"] += 1
                    continue
                stats["characters"] += len(text)
                stats["utf8_bytes"] += len(text.encode("utf-8"))
                count += 1
                yield text
        print(f"Read {path.name}: {count} nonempty selected books", flush=True)


def new_stats():
    return dict(books=0, duplicate_rows_skipped=0, headers_removed=0, footers_removed=0, empty_books=0, characters=0, utf8_bytes=0)


def main():
    files = sorted(DATA.glob("en-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No English Parquet files in {DATA}")
    ids = []
    for path in files:
        ids.extend(pq.read_table(path, columns=["id"]).column("id").to_pylist())
    train_ids, test_ids = split_books(ids)
    # Refuse to overwrite a vocabulary whose IDs may already be used by a model.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifest = {"split_method": "unique Gutenberg IDs before encoding suffix sorted by SHA256('42:' + book_id), first floor(80%) train", "duplicate_policy": "first occurrence per base book ID in sorted shard and row order", "files": [p.name for p in files], "train": train_ids, "test": test_ids}
    (OUTPUT / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Split: {len(train_ids)} train / {len(test_ids)} test books", flush=True)
    train_stats = new_stats()
    started = time.monotonic()
    tokenizer.train_from_iterator(books(files, set(train_ids), train_stats), trainer=trainer, length=len(train_ids))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError(f"Expected {VOCAB_SIZE} tokens, got {tokenizer.get_vocab_size()}")
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.token_to_id(token) != expected_id:
            raise ValueError(f"Unexpected special token ID for {token}")
    tokenizer.save(str(OUTPUT / "tokenizer.json"))
    trained = Tokenizer.from_file(str(OUTPUT / "tokenizer.json"))
    training_seconds = time.monotonic() - started
    test_stats = new_stats()
    test_tokens = 0
    unknown_tokens = 0
    unk_id = trained.token_to_id("<unk>")
    for text in books(files, set(test_ids), test_stats):
        encoding = trained.encode(text)
        if trained.decode(encoding.ids, skip_special_tokens=False) != text:
            raise ValueError("Held-out encode/decode round trip failed")
        test_tokens += len(encoding.ids)
        unknown_tokens += encoding.ids.count(unk_id)
    for text in ["Hello, world!\n  Two spaces.", "Café — naïve 🙂", "<pad><unk><bos><eos>"]:
        assert trained.decode(trained.encode(text).ids, skip_special_tokens=False) == text
    report = {"vocab_size": trained.get_vocab_size(), "special_token_ids": {t: trained.token_to_id(t) for t in SPECIAL_TOKENS}, "training_seconds": training_seconds, "total_seconds": time.monotonic() - started, "train": train_stats, "test": test_stats, "test_tokens": test_tokens, "test_unknown_tokens": unknown_tokens, "test_characters_per_token": test_stats["characters"] / test_tokens, "all_nonempty_test_books_roundtrip": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
