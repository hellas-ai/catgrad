import sys
import argparse
import torch
from transformers import (
    logging,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.image_utils import load_image

torch.set_printoptions(linewidth=200)

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
    parser.add_argument("-i", "--image", type=str, default=None)
    parser.add_argument("-r", "--raw", action="store_true")
    parser.add_argument("-t", "--thinking", action="store_true")
    parser.add_argument("--cache", dest="use_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-d", "--dtype", type=str, default="float32")
    args = parser.parse_args()

    if args.image is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, revision=args.revision, dtype=args.dtype
            )
        except:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model, revision=args.revision, dtype=args.dtype
            )
    else:
        processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, revision=args.revision, dtype=args.dtype
        )

    print(f"Loaded model {args.model}, dtype:{model.dtype} on device {model.device}", file=sys.stderr)
    prompt = args.prompt

    if args.image is None and not args.raw and tokenizer.chat_template is not None:
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

    if args.image is None:
        inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt")
        logits = model.generate(**inputs, max_new_tokens=args.seq_len, do_sample=False, use_cache=args.use_cache)
        output = tokenizer.decode(logits[0], skip_special_tokens=True)
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "path": args.image},
                ],
            }
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                do_image_splitting=False,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            image = load_image(args.image)
            inputs = processor(images=image, text=prompt, return_tensors="pt")

        inputs = inputs.to(model.device, dtype=model.dtype)
        input_len = inputs["input_ids"].shape[-1]
        logits = model.generate(**inputs, max_new_tokens=args.seq_len, do_sample=False, use_cache=args.use_cache)
        output = processor.decode(logits[0][input_len:], skip_special_tokens=True)

    print(output)
