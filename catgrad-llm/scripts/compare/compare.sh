#!/usr/bin/env bash
set -euo pipefail

# List of models to test in CI
MODELS=(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
    "openai-community/gpt2"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3.5-0.8B"
    "ibm-granite/granite-3.1-1b-a400m-instruct"
    "LiquidAI/LFM2-350M"
)

# Add more models if not in GitHub CI (they are larger and/or need a user agreement to download from HF)
if [[ -z "${GITHUB_ACTIONS:-}" ]]; then
    MODELS+=(
        "meta-llama/Llama-3.2-1B-Instruct"
        "google/gemma-3-270m-it"
        "allenai/OLMo-2-0425-1B-Instruct"
    )
fi

DIR=$(dirname "$0")
REFERENCE_DIR="$DIR/expected"

MAXLEN="${CATGRAD_COMPARE_MAXLEN:-40}"
OUTPUT_DIR=$DIR/outputs/$MAXLEN
mkdir -p "$OUTPUT_DIR"
rm -rf "$OUTPUT_DIR"/*

echo "Generating outputs of ${MAXLEN} tokens for ${#MODELS[@]} models..."

if [[ "${CATGRAD_COMPARE_HF_RUN:-}" ]]; then
    REFERENCE_DIR=$DIR/expected/$MAXLEN
    mkdir -p $REFERENCE_DIR

    for model in "${MODELS[@]}"; do
        # Replace slashes with dashes for the filename
        filename="${model//\//-}"

        echo "Running HF Transformers for $model -> $REFERENCE_DIR/$filename"

        uv run catgrad-llm/scripts/llm.py -m "$model" -p 'Category theory is' -s $MAXLEN -r > "$REFERENCE_DIR/$filename" 2>/dev/null &
    done

    wait
fi


for model in "${MODELS[@]}"; do
    # Replace slashes with dashes for the filename
    filename="${model//\//-}"
    
    echo "Running for $model -> $OUTPUT_DIR/$filename"

    TYPECHECK="-t"

    ./target/release/examples/llama -m "$model" -p 'Category theory is' -s $MAXLEN --raw -k $TYPECHECK > "$OUTPUT_DIR/$filename" 2>/dev/null &

    [[ -z "${GITHUB_ACTIONS:-}" ]] || wait
done

wait

echo "Comparing $OUTPUT_DIR with $REFERENCE_DIR..."

if diff -ur "$OUTPUT_DIR" "$REFERENCE_DIR"; then
    echo "Success: All model outputs match."
    exit 0
else
    echo "Failure: Differences found in model outputs."
    exit 1
fi
