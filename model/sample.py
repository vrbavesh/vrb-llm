"""Generate text from a trained checkpoint.

Loads the architecture recorded inside the checkpoint rather than the current
`ARCH`, so an older checkpoint keeps generating with the geometry it was trained
with. Generation stops at <eos> and special tokens are stripped from the output.

    .\\.venv\\Scripts\\python.exe -m model.sample --prompt "The old man walked"
"""

import argparse
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer

# Allow `python model\sample.py` as well as `python -m model.sample`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.gpt import GPT, GPTConfig


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer" / "artifacts" / "tokenizer.json"


def load(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**checkpoint["arch"])
    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, checkpoint["step"]


def generate(model, cfg, tokenizer, prompt, new_tokens, temperature, top_k, device):
    if prompt:
        ids = tokenizer.encode(prompt).ids
    else:
        ids = [tokenizer.token_to_id("<bos>")]

    room = cfg.max_seq_len - len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model.generate(idx, min(new_tokens, room), temperature=temperature, top_k=top_k)

    eos = tokenizer.token_to_id("<eos>")
    produced = []
    # Slice off the prompt: out[0] is prompt + continuation, and the caller prints
    # the prompt itself, so decoding it here would duplicate it.
    for token in out[0].tolist()[len(ids):]:
        if token == eos:
            break
        produced.append(token)
    return tokenizer.decode(produced, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model" / "runs" / "main" / "last.pt")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", type=Path, help="also write the output here as UTF-8")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    model, cfg, step = load(args.checkpoint, device)
    torch.manual_seed(args.seed)

    lines = []
    lines.append(f"checkpoint: {args.checkpoint.name}  step {step}  "
                 f"params {model.num_params():,}  temperature {args.temperature}  top-k {args.top_k}")
    lines.append(f"prompt: {args.prompt!r}")
    for index in range(args.samples):
        text = generate(model, cfg, tokenizer, args.prompt, args.tokens,
                        args.temperature, args.top_k, device)
        lines.append(f"--- sample {index + 1} ---")
        lines.append(args.prompt + text if args.prompt else text)
        lines.append("")

    payload = "\n".join(lines)
    print(payload)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
