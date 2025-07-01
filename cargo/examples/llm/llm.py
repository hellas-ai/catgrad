import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="openai-community/gpt2",
    )
    parser.add_argument("-p", "--prompt", type=str, default="Hello world")
    parser.add_argument("-s", "--seq-len", type=int, default=10)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Remove model settings so generate does not warn about sampling parameters
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    inputs = tokenizer(args.prompt, return_tensors="pt")
    logits = model.generate(**inputs, max_new_tokens=args.seq_len, do_sample=False)
    output = tokenizer.decode(logits[0])

    print(output)
