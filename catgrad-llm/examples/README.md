## Testing llm inference with Catgrad

These examples work with safetensor weights as found on [Huggingface Hub](https://huggingface.co/models)

### Supported architectures: ###

**GPT-2**, **Llama-3**, **Qwen-3**, **OLMo-2**, **Gemma-3**, **Phi-3**, **SmolLM3-3B**, **Granite-3**

### LLM example ###

The `llm` example uses `model.safetensors`, `tokenizer.json`, `tokenizer_config.json` and `config.json` files from models under  `~/.cache/huggingface/hub/`.
It either downloads the files or reuses the ones already in the cache (maybe previously downloaded by other frameworks like Candle, Transformers or vLLM).

```
cargo run --release --example llm -- -m openai-community/gpt2 -p 'Category theory' -s 9
Category theory is a theory of how the universe works.
```

The `llm.py` script can be used to compare the outputs to those generated by Huggingface Transformers
```
python examples/llm/llm.py -m openai-community/gpt2 -p 'Category theory' -s 9
Category theory is a theory of how the universe works.
```

For chat/instruct tuned models a chat template will be read from `tokenizer_config.json` and the prompt will be formatted accordingly.
Pass -r for raw prompts without chat template application.

Here are links to some supported models to test the `llm` example with. All are chat models except GPT2.

<https://huggingface.co/openai-community/gpt2>

<https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>

<https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>

<https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct>

<https://huggingface.co/HuggingFaceTB/SmolLM3-3B>

<https://huggingface.co/Qwen/Qwen3-0.6B>

<https://huggingface.co/google/gemma-3-1b-it>

<https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>

<https://huggingface.co/microsoft/Phi-4-mini-instruct>

<https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct>

<https://huggingface.co/ibm-granite/granite-3.3-2b-instruct>
