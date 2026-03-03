#!/usr/bin/env bash
set -euo pipefail

# List of models to test
MODELS=(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
    "openai-community/gpt2"
    "Qwen/Qwen3-0.6B"
    "google/gemma-3-270m-it"
)

DIR=$(dirname "$0")
REFERENCE_DIR="$DIR/expected"
OUTPUT_DIR="$DIR/outputs"

mkdir -p "$OUTPUT_DIR"

echo "Generating outputs for ${#MODELS[@]} models..."

for model in "${MODELS[@]}"; do
    # Replace slashes with dashes for the filename
    filename="${model//\//-}"
    
    echo "Running for $model -> $OUTPUT_DIR/$filename"
   
    cargo run -r --example llama -- -m "$model" -p 'Category theory is' -s 40 --raw -k > "$OUTPUT_DIR/$filename" 2>/dev/null
done

echo "Comparing $OUTPUT_DIR with $REFERENCE_DIR..."

if diff -ur "$OUTPUT_DIR" "$REFERENCE_DIR"; then
    echo "Success: All model outputs match."
    exit 0
else
    echo "Failure: Differences found in model outputs."
    exit 1
fi
