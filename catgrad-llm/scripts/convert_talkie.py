"""Convert a Talkie release into a catgrad-loadable HF-style directory.

Input:  a Talkie release as published on HuggingFace
        (https://huggingface.co/talkie-lm/talkie-1930-13b-{base,it}):
          - final.ckpt / rl-refined.pt / base.ckpt   (raw torch.save dict)
          - vocab.txt                                 (tiktoken BPE)

Output: a directory catgrad-llm's loader can consume:
          - model.safetensors        (bf16 weights, talkie-native key names)
          - config.json              ({"architectures": ["TalkieForCausalLM"], ...})
          - tokenizer.json           (HF tokenizers, byte-level BPE)
          - tokenizer_config.json    (chat template for IT variants)

Dependencies: torch, safetensors, tiktoken, tokenizers.

Usage:
    python convert_talkie.py \
        --ckpt /path/to/final.ckpt \
        --vocab /path/to/vocab.txt \
        --out /path/to/talkie-1930-13b-base \
        --style base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from tiktoken.load import load_tiktoken_bpe
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers


BASE_VOCAB_SIZE = 65536

# Matches src/talkie/tokenizer.py — the tiktoken pat_str, joined with `|`.
PAT_STR = "|".join(
    [
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

BASE_SPECIAL_TOKENS = {"<|endoftext|>": BASE_VOCAB_SIZE - 1}
IT_SPECIAL_TOKENS = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
    "<|end|>": BASE_VOCAB_SIZE,
    "<|user|>": BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>": BASE_VOCAB_SIZE + 3,
}

CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "<|{{ message.role }}|>{{ message.content }}<|end|>"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def convert_weights(ckpt_path: Path, out_path: Path, dtype: torch.dtype) -> dict:
    """Pickle → safetensors. Preserves Talkie's native param names verbatim
    (`embed.weight`, `blocks.{i}.attn.attn_query.weight`, …, `lm_head`,
    `lm_head_gain.w_g`); strips `_orig_mod.` from torch.compile.

    Returns the inferred config dict so the caller can write `config.json`.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    else:
        sd = raw
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    # Sanity check: the keys we expect.
    required = {"embed.weight", "lm_head", "lm_head_gain.w_g"}
    missing = required - sd.keys()
    if missing:
        raise SystemExit(f"checkpoint missing keys: {sorted(missing)}")

    vocab_size, n_embd = sd["embed.weight"].shape
    n_layer = 1 + max(
        int(k.split(".")[1]) for k in sd if k.startswith("blocks.")
    )
    n_head = sd["blocks.0.attn.head_gain.head_g"].shape[0]
    head_dim = n_embd // n_head

    # Cast in place to halve peak memory: 13B fp32 + 13B bf16 ≈ 80 GB if held
    # together; popping each fp32 tensor as we cast keeps it near 30 GB.
    converted = {}
    for k in list(sd.keys()):
        converted[k] = sd.pop(k).to(dtype).contiguous()
    save_file(converted, out_path)

    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(n_embd),
        "num_hidden_layers": int(n_layer),
        "num_attention_heads": int(n_head),
        "head_dim": int(head_dim),
    }


def write_config(out_dir: Path, weights_meta: dict, style: str) -> None:
    """Synthesize config.json. Talkie has no native config — everything
    is hardcoded in src/talkie/model.py's GPTConfig + the call to
    `_precompute_rotary_embeddings(base=1_000_000)`.
    """
    eos_id = (
        IT_SPECIAL_TOKENS["<|end|>"] if style == "it"
        else BASE_SPECIAL_TOKENS["<|endoftext|>"]
    )
    config = {
        "architectures": ["TalkieForCausalLM"],
        "model_type": "talkie",
        **weights_meta,
        "max_position_embeddings": 2048,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,  # F.rms_norm in talkie uses dtype-default; small ε is closest in fp32
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "eos_token_id": eos_id,
    }
    if style == "it":
        # IT model expanded the embedding to 65540 during fine-tuning.
        config["vocab_size"] = max(config["vocab_size"], BASE_VOCAB_SIZE + 4)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte→unicode mapping. Reversibly maps every byte to a printable
    Unicode codepoint so byte-level BPE can be expressed as a string-keyed
    vocab + merges file."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _bpe_split(token: bytes, ranks: dict[bytes, int]) -> tuple[bytes, bytes]:
    """Recover the (left, right) merge that produced `token` by replaying the
    BPE algorithm with all ranks strictly less than `ranks[token]`.
    Standard recipe — same one transformers' TikTokenConverter uses."""
    parts = [bytes([b]) for b in token]
    target = ranks[token]
    while True:
        min_idx, min_rank = None, None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < target and (min_rank is None or r < min_rank):
                min_idx, min_rank = i, r
        if min_idx is None:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
    if len(parts) != 2:
        raise ValueError(f"could not reduce token {token!r} (rank {target}) to a 2-part merge")
    return parts[0], parts[1]


def write_tokenizer(vocab_path: Path, out_dir: Path, style: str) -> None:
    """Convert tiktoken vocab.txt → HF tokenizer.json (byte-level BPE)."""
    ranks = load_tiktoken_bpe(str(vocab_path))
    # Talkie drops the highest rank (reserved for <|endoftext|>).
    ranks = {tok: r for tok, r in ranks.items() if r < BASE_VOCAB_SIZE - 1}

    b2u = _bytes_to_unicode()

    def encode(b: bytes) -> str:
        return "".join(b2u[x] for x in b)

    vocab = {encode(tok): r for tok, r in ranks.items()}
    merges = []
    for tok, _ in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(tok) == 1:
            continue
        left, right = _bpe_split(tok, ranks)
        merges.append((encode(left), encode(right)))

    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(PAT_STR), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tk.decoder = decoders.ByteLevel()

    specials = IT_SPECIAL_TOKENS if style == "it" else BASE_SPECIAL_TOKENS
    tk.add_special_tokens(list(specials.keys()))

    tk.save(str(out_dir / "tokenizer.json"))

    tcfg = {
        "model_max_length": 2048,
        "bos_token": None,
        "eos_token": "<|end|>" if style == "it" else "<|endoftext|>",
        "pad_token": None,
        "added_tokens_decoder": {
            str(idx): {"content": tok, "special": True}
            for tok, idx in specials.items()
        },
    }
    if style == "it":
        tcfg["chat_template"] = CHAT_TEMPLATE
    (out_dir / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2))


def verify_tokenizer(vocab_path: Path, out_dir: Path, style: str) -> None:
    """Round-trip a sample through both tiktoken and the converted HF tokenizer.
    Failures here mean the BPE reconstruction or pre-tokenizer regex diverged —
    do not ship without this passing."""
    import tiktoken

    ranks = load_tiktoken_bpe(str(vocab_path))
    ranks = {tok: r for tok, r in ranks.items() if r < BASE_VOCAB_SIZE - 1}
    specials = IT_SPECIAL_TOKENS if style == "it" else BASE_SPECIAL_TOKENS
    tt = tiktoken.Encoding(
        name="talkie", pat_str=PAT_STR, mergeable_ranks=ranks, special_tokens=specials
    )
    hf = Tokenizer.from_file(str(out_dir / "tokenizer.json"))

    samples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "It's a fine day in 1930.\nLet us discuss aeronautics.",
        "  leading spaces and\ttabs\n\nand newlines",
        "café naïve résumé — em-dash and ellipsis…",
    ]
    if style == "it":
        samples.append("<|user|>hi<|end|><|assistant|>")

    for s in samples:
        a = tt.encode(s, allowed_special=set(specials))
        b = hf.encode(s).ids
        if a != b:
            raise SystemExit(
                f"tokenizer mismatch on {s!r}\n  tiktoken: {a}\n  hf:       {b}"
            )
    print(f"tokenizer round-trip ok ({len(samples)} samples)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", choices=["base", "it"], required=True)
    ap.add_argument("--dtype", choices=DTYPES.keys(), default="bf16")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("converting weights…")
    meta = convert_weights(args.ckpt, args.out / "model.safetensors", DTYPES[args.dtype])

    print("writing config.json…")
    write_config(args.out, meta, args.style)

    print("writing tokenizer.json + tokenizer_config.json…")
    write_tokenizer(args.vocab, args.out, args.style)

    print("verifying tokenizer round-trip…")
    verify_tokenizer(args.vocab, args.out, args.style)

    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
