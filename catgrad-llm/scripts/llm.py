import argparse
import json
import re
import sys
from typing import Literal

import torch
from transformers.image_utils import load_image

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    logging,
)

torch.set_printoptions(linewidth=200)

logging.set_verbosity_error()


def calculator(
    left: float,
    right: float,
    operator: Literal["add", "subtract", "multiply", "divide"],
) -> float:
    """
    Calculate a result from two numbers.

    Args:
        left: The left-hand number.
        right: The right-hand number.
        operator: The operation to apply.

    Returns:
        The calculated result.
    """
    if operator == "add":
        return left + right
    if operator == "subtract":
        return left - right
    if operator == "multiply":
        return left * right
    if right == 0:
        raise ValueError("division by zero")
    return left / right


def parse_scalar(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_tool_call(tool_call):
    if "name" not in tool_call:
        return None

    arguments = tool_call.get("arguments", tool_call.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None

    return {"name": tool_call["name"], "arguments": arguments}


def parse_tool_call(text):
    "Parse Qwen-3 JSON and Qwen-3.5 XML tool calls from the model output."
    if match := re.search(
        r"<tool_call>\s*<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
        text,
        flags=re.DOTALL,
    ):
        arguments = {
            parameter.group("name"): parse_scalar(parameter.group("value"))
            for parameter in re.finditer(
                r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
                match.group("body"),
                flags=re.DOTALL,
            )
        }
        return {"name": match.group("name"), "arguments": arguments}

    if match := re.search(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        text,
        flags=re.DOTALL,
    ):
        try:
            return normalize_tool_call(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return normalize_tool_call(json.loads(stripped))
        except json.JSONDecodeError:
            return None

    return None


def generate_chat(tokenizer, model, messages, args, tools=None):
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    logits = model.generate(
        **inputs,
        max_new_tokens=args.seq_len,
        do_sample=False,
        use_cache=args.use_cache,
    )
    return tokenizer.decode(logits[0][input_len:], skip_special_tokens=True)


def run_tool_chat(tokenizer, model, prompt, args):
    tools = [calculator]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the calculator tool for arithmetic questions.",
        },
        {"role": "user", "content": prompt},
    ]
    output = generate_chat(tokenizer, model, messages, args, tools=tools)
    # print(output)
    tool_call = parse_tool_call(output)
    if tool_call is None:
        return output

    if tool_call["name"] != calculator.__name__:
        raise ValueError(f"unsupported tool call: {tool_call['name']}")

    tool_call_id = "call_1"
    tool_content = json.dumps({"result": calculator(**tool_call["arguments"])})
    print(
        f"Tool {tool_call['name']}({tool_call['arguments']}) -> {tool_content}",
        file=sys.stderr,
    )
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_call["name"],
                "content": tool_content,
            },
        ]
    )
    return generate_chat(tokenizer, model, messages, args, tools=tools)


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
    parser.add_argument(
        "--tool-use",
        action="store_true",
        help="Enable a simple calculator tool flow.",
    )
    parser.add_argument(
        "--cache", dest="use_cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("-d", "--dtype", type=str, default="float32")
    args = parser.parse_args()

    if args.tool_use and args.image is not None:
        parser.error("--tool-use does not support --image")

    if args.tool_use and args.raw:
        parser.error("--tool-use does not support --raw")

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

    print(
        f"Loaded model {args.model}, dtype:{model.dtype} on device {model.device}",
        file=sys.stderr,
    )
    prompt = args.prompt

    if (
        args.image is None
        and not args.raw
        and not args.tool_use
        and tokenizer.chat_template is not None
    ):
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
        if args.tool_use:
            output = run_tool_chat(tokenizer, model, prompt, args)
        else:
            inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt")
            logits = model.generate(
                **inputs,
                max_new_tokens=args.seq_len,
                do_sample=False,
                use_cache=args.use_cache,
            )
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
        logits = model.generate(
            **inputs,
            max_new_tokens=args.seq_len,
            do_sample=False,
            use_cache=args.use_cache,
        )
        output = processor.decode(logits[0][input_len:], skip_special_tokens=True)

    print(output)
