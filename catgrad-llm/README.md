# Catgrad LLM

LLMs in Catgrad.

The intent is to have three main self-contained modules:

1. `models` allow *building* LLM graphs: the core tensor network defining the model
2. `run` (WIP) is for *running* LLM graphs given a 'package' of additional information like weights and tokenizers.
    - A stateful interface managing interpreters & KV caches etc.
    - Manages packaging configuration & weights from e.g. huggingface
3. `types` are abstract serving interfaces plus API wire-format types

This is not reflected by the current state of the code.
What's missing:

1. A definition of the "package" of supporting information in `run`
2. Shape polymorphism in graph definitions (so graphs don't need to be rebuilt every run)

Future changes: split `run` into generic "tensor runtime" and "llm code" where:
    - tensor runtime: (run graph with arrays only; state/cache aware, tokenization-*un*aware, runnable on remote host)
    - LLM code: tokenization, chat templates, etc.
