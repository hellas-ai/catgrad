import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from transformers import logging

logging.set_verbosity_error()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-community/gpt2",
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("-p", "--prompt", type=str, default="Category theory is")
    parser.add_argument("-s", "--seq-len", type=int, default=10)
    parser.add_argument("-r", "--raw-prompt", action="store_true")
    parser.add_argument("-t", "--thinking", action="store_true")
    parser.add_argument("-d", "--dtype", type=str, default="float32")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=args.dtype
        )
    except:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, revision=args.revision, dtype=args.dtype
        )

    print(f"Loaded model {args.model}, dtype:{model.dtype}")
    prompt = args.prompt

    if not args.raw_prompt and tokenizer.chat_template is not None:
        chat = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.thinking,
        )

    # Remove model settings so generate does not warn about sampling parameters
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt")
    logits = model.generate(**inputs, max_new_tokens=args.seq_len, do_sample=False)
    output = tokenizer.decode(logits[0])

    print(output)
