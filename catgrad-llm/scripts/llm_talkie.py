"""Greedy-decode reference for Talkie. Mirrors `scripts/llm.py`'s
`do_sample=False` path but uses the talkie package (talkie isn't in
HF transformers, so AutoModelForCausalLM doesn't apply).

Output is true argmax (no Gumbel sampling, no temperature) so the
result is byte-identical across runs and directly comparable to
catgrad-llm's argmax-greedy decode.

Two loader paths:

  * `--ckpt`         — talkie's native `final.ckpt`. Matches upstream
                      load semantics, but `torch.load` materialises the
                      whole 53 GB fp32 dict in RAM before casting to
                      bf16 (transient peak ≈ 100 GB).
  * `--safetensors`  — pre-converted bf16 safetensors (output of
                      `convert_talkie.py`). Peak ≈ 52 GB.

Both produce bit-identical model state once cast, so use safetensors
for the stability harness.

Two run modes:

  * single-shot — `--prompt P --seq-len N`, output goes to `--out` or stdout.
  * batch       — `--batch`, reads JSON-line cases from stdin
                  (`{"prompt": "...", "seq_len": N, "out": "/path"}`),
                  loads the model once, runs all cases. The matrix harness
                  uses this so we don't pay the 50 GB load N times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from talkie.model import GPTConfig, TalkieModel, resize_model_embeddings
from talkie.tokenizer import IT_VOCAB_SIZE, build_tokenizer


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_from_safetensors(
    path: Path, device: torch.device, target_vocab_size: int | None
) -> TalkieModel:
    """Build a TalkieModel from our pre-converted bf16 safetensors.

    Skips the ckpt → fp32-dict → cast round-trip in talkie's own loader,
    which would otherwise spike to ~100 GB on a 13B model.
    """
    from safetensors.torch import load_file

    sd = load_file(str(path), device="cpu")
    vocab_size = sd["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=vocab_size)

    cpu = torch.device("cpu")
    model = TalkieModel(config, cpu)
    model.load_state_dict(sd, strict=True)
    del sd

    if target_vocab_size is not None and vocab_size < target_vocab_size:
        model = resize_model_embeddings(model, target_vocab_size, cpu)

    model = model.to(dtype=torch.bfloat16).to(device)
    model.device = device
    model.eval()
    return model


def load_model(args, device: torch.device) -> TalkieModel:
    target_vocab = IT_VOCAB_SIZE if args.style == "it" else None
    if args.safetensors is not None:
        model = load_from_safetensors(args.safetensors, device, target_vocab)
    else:
        from talkie.model import load_checkpoint
        model = load_checkpoint(str(args.ckpt), device, target_vocab_size=target_vocab)
    if DTYPES[args.dtype] != torch.bfloat16:
        model = model.to(dtype=DTYPES[args.dtype])
    model.eval()
    return model


def generate(model: TalkieModel, tokenizer, prompt: str, seq_len: int) -> str:
    ids = tokenizer.encode(prompt, allowed_special="all")
    x = torch.tensor([ids], device=model.device, dtype=torch.long)
    with torch.no_grad():
        for _ in range(seq_len):
            logits = model.forward(x)
            next_id = int(torch.argmax(logits, dim=-1).item())
            x = torch.cat(
                [x, torch.tensor([[next_id]], device=model.device, dtype=torch.long)],
                dim=1,
            )
    return tokenizer.decode(x[0].tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="talkie native final.ckpt (slow load)")
    src.add_argument("--safetensors", type=Path, help="bf16 safetensors (fast load)")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--style", choices=["base", "it"], default="base")
    ap.add_argument("--dtype", choices=DTYPES.keys(), default="bf16")
    ap.add_argument("--device", default=None,
                    help="Defaults to cuda if available, otherwise cpu.")
    ap.add_argument("-p", "--prompt", help="single-shot prompt")
    ap.add_argument("-s", "--seq-len", type=int, default=20)
    ap.add_argument("-o", "--out", type=Path, help="single-shot output file (default stdout)")
    ap.add_argument("--batch", action="store_true",
                    help="Read JSONL cases from stdin (one {prompt, seq_len, out} per line).")
    args = ap.parse_args()

    if args.batch and args.prompt is not None:
        ap.error("--batch and --prompt are mutually exclusive")
    if not args.batch and args.prompt is None:
        ap.error("either --batch or --prompt is required")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading talkie on {device} dtype={args.dtype}…", file=sys.stderr, flush=True)
    tokenizer = build_tokenizer(str(args.vocab), style=args.style)
    model = load_model(args, device)
    print("loaded.", file=sys.stderr, flush=True)

    if args.batch:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            case = json.loads(line)
            text = generate(model, tokenizer, case["prompt"], int(case["seq_len"]))
            Path(case["out"]).write_text(text + "\n")
            print(f"  case {case['prompt']!r} s={case['seq_len']} → {case['out']}",
                  file=sys.stderr, flush=True)
    else:
        text = generate(model, tokenizer, args.prompt, args.seq_len)
        if args.out:
            args.out.write_text(text + "\n")
        else:
            print(text)


if __name__ == "__main__":
    main()
