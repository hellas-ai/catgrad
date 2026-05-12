# Catgrad LLM Models #

This crate allows building model graphs.
Running them requires an interpreter/runtime, such as the catgrad-llm crate.

### Model dump example ###

The `dump_model` example fetches only `config.json`, builds a text-model graph, and saves it to JSON.
It does not depend on a runtime, only on the model files.

```
cargo run --release --example dump_model -- -m lfm -s 256 --dump lfm.json
```

Load and run a previously dumped graph:
```
cargo run --release --example llama -- --load lfm.json -m lfm -p 'Category theory' -s 20
```

