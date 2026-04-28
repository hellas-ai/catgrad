#!/usr/bin/env bash
# Stability harness for the Talkie port.
#
# Token-by-token diffs the catgrad implementation against the talkie
# package's PyTorch reference. The reference loader holds the model in
# RAM for the whole matrix (one load, all cases) — without that, each
# 13B Python process spikes to ~50 GB and back-to-back runs OOM.
#
# Three outputs per case:
#   ref      — talkie / pytorch  (CPU, dtype-cast bf16 by default)
#   cat-k    — catgrad-llm with KV cache
#   cat-nok  — catgrad-llm without KV cache
#
# Comparisons:
#   ref vs cat-k     — does the port match the upstream reference?
#                      This is the headline correctness check.
#   cat-k vs cat-nok — does catgrad's cache implementation match its
#                      uncached path? In bf16 this can drift by one
#                      argmax tie-break and is informative but not
#                      strictly a talkie-level concern.
#
# Modes:
#   * Single-shot — set TALKIE_PROMPT or TALKIE_SEQLEN; runs that one case.
#   * Matrix     — default. Built-in suite of prompts × lengths.
#
# Required env:
#   TALKIE_DIR        directory with the converted model (config.json,
#                     model.safetensors, tokenizer.json, ...)
#   TALKIE_VOCAB      path to the original vocab.txt
#   TALKIE_VENV       path to a venv with the `talkie` package installed
#                     (`pip install git+https://github.com/talkie-lm/talkie`)
#
# Optional env:
#   TALKIE_STYLE      base | it (default: base)
#   TALKIE_PROMPT     single-shot override
#   TALKIE_SEQLEN     single-shot override
#   TALKIE_DTYPE      bf16 | fp16 | fp32 (default: bf16)
#   TALKIE_CKPT       fall back to talkie's native ckpt loader (slow,
#                     transient ~100 GB RAM peak). Default is to load
#                     bf16 safetensors from $TALKIE_DIR/model.safetensors.
#   TALKIE_INCLUDE_NOCACHE=1
#                     also run catgrad without KV cache and diff. This
#                     tests catgrad-internal consistency rather than
#                     talkie correctness; in bf16 the no-cache path
#                     drifts by one argmax tie-break for some prompts
#                     and is much slower (O(N²) per step). Off by default.

set -euo pipefail

: "${TALKIE_DIR:?set TALKIE_DIR to the converted model directory}"
: "${TALKIE_VOCAB:?set TALKIE_VOCAB to the original vocab.txt path}"
: "${TALKIE_VENV:?set TALKIE_VENV to a python venv with talkie installed}"

STYLE="${TALKIE_STYLE:-base}"
DTYPE="${TALKIE_DTYPE:-bf16}"
SAFETENSORS="$TALKIE_DIR/model.safetensors"

DIR=$(dirname "$0")
SCRIPTS=$(cd "$DIR/.." && pwd)
WORKSPACE=$(cd "$SCRIPTS/../.." && pwd)
LLAMA_BIN="$WORKSPACE/target/release/examples/llama"

[[ -x "$LLAMA_BIN" ]] || {
  echo "build the example first:" >&2
  echo "  cargo build --release --features metal --example llama -p catgrad-llm" >&2
  exit 1
}

# Build the test-case matrix. Each line is a JSON object emitted to a
# temp file. The Python ref process consumes them as JSONL, the bash
# loop iterates the same list for catgrad runs and diffing.
build_cases() {
  local cases_jsonl="$1"
  local out_dir="$2"
  : > "$cases_jsonl"
  for prompt in "${PROMPTS[@]}"; do
    for seqlen in "${SEQLENS[@]}"; do
      local key
      key=$(printf '%s' "${prompt}__${seqlen}" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)
      python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'seq_len':int(sys.argv[2]),'out':sys.argv[3]}))" \
        "$prompt" "$seqlen" "$out_dir/ref__$key" >> "$cases_jsonl"
    done
  done
}

run_python_ref_batch() {
  local cases_jsonl="$1" loader_args
  if [[ -n "${TALKIE_CKPT:-}" ]]; then
    loader_args=(--ckpt "$TALKIE_CKPT")
  else
    loader_args=(--safetensors "$SAFETENSORS")
  fi
  # shellcheck disable=SC1091
  source "$TALKIE_VENV/bin/activate"
  python "$SCRIPTS/llm_talkie.py" \
    "${loader_args[@]}" --vocab "$TALKIE_VOCAB" --style "$STYLE" \
    --dtype "$DTYPE" --batch < "$cases_jsonl"
}

cat_dtype() {
  # Python side uses bf16/fp16/fp32; the llama example uses bf16/f16/f32.
  case "$1" in
    fp32) echo "f32" ;;
    fp16) echo "f16" ;;
    *)    echo "$1" ;;
  esac
}

run_cat() {
  local prompt="$1" seqlen="$2" out="$3" extra_flags="$4"
  local cat_dt; cat_dt=$(cat_dtype "$DTYPE")
  # shellcheck disable=SC2086
  if ! "$LLAMA_BIN" -m "$TALKIE_DIR" --raw $extra_flags --dtype "$cat_dt" \
       -p "$prompt" -s "$seqlen" > "$out" 2>"$out.err"; then
    echo "[cat] llama failed for prompt=$prompt seqlen=$seqlen:" >&2
    sed 's/^/  /' < "$out.err" >&2
    return 1
  fi
}

# Compare two outputs token-by-token (using the talkie tokenizer so
# token boundaries are real, not whitespace). Prints one of:
#
#   ok                     — full match
#   drift TOK/TOTAL CTX    — diverged at token TOK out of TOTAL; CTX shows
#                            the first ref-vs-cat tokens after the split
#
# Used both to call PASS/FAIL and to surface the matched-prefix length,
# which is the actual signal in cross-implementation bf16: 100% prefix
# match is "exact"; partial match quantifies how far before noise flips
# a borderline argmax.
diff_summary() {
  local a="$1" b="$2"
  TALKIE_VOCAB="$TALKIE_VOCAB" TALKIE_STYLE="$STYLE" \
  "$TALKIE_VENV/bin/python" - "$a" "$b" <<'PY'
import os, pathlib, sys
from talkie.tokenizer import build_tokenizer
tk = build_tokenizer(os.environ["TALKIE_VOCAB"], style=os.environ.get("TALKIE_STYLE", "base"))
a = pathlib.Path(sys.argv[1]).read_text().rstrip("\n")
b = pathlib.Path(sys.argv[2]).read_text().rstrip("\n")
ta = tk.encode(a, allowed_special="all")
tb = tk.encode(b, allowed_special="all")
n = min(len(ta), len(tb))
i = 0
while i < n and ta[i] == tb[i]:
    i += 1
total = max(len(ta), len(tb))
if i == len(ta) == len(tb):
    print("ok")
else:
    a_tail = tk.decode(ta[i : i + 5])
    b_tail = tk.decode(tb[i : i + 5])
    print(f"drift {i}/{total}  ref={a_tail!r} cat={b_tail!r}")
PY
}

run_matrix() {
  local out_dir cases_jsonl
  out_dir=$(mktemp -d)
  cases_jsonl="$out_dir/cases.jsonl"
  trap 'rm -rf "$out_dir"' RETURN

  build_cases "$cases_jsonl" "$out_dir"

  echo "Talkie stability matrix — dtype=$DTYPE style=$STYLE"
  echo "ref:    $([[ -n "${TALKIE_CKPT:-}" ]] && echo "$TALKIE_CKPT" || echo "$SAFETENSORS")"
  echo "cat:    $TALKIE_DIR"
  echo "cases:  $(wc -l <"$cases_jsonl") (${#PROMPTS[@]} prompts × ${#SEQLENS[@]} lens)"
  echo

  # Phase 1: Python ref. One load, all cases. Slowest part.
  echo "[ref] generating all reference outputs (one model load) …"
  local t0; t0=$(date +%s)
  run_python_ref_batch "$cases_jsonl"
  echo "[ref] done in $(( $(date +%s) - t0 ))s"
  echo

  # Phase 2: catgrad cat-k for every case. Optional cat-nok if requested
  # (it's much slower and is a catgrad-internal consistency check, not a
  # talkie-correctness one).
  echo "[cat] running catgrad cached$([[ -n "${TALKIE_INCLUDE_NOCACHE:-}" ]] && echo " + uncached") for each case …"
  local total_matched=0 total_tokens=0
  while IFS= read -r line; do
    local prompt seqlen ref out_k out_nok
    prompt=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['prompt'])" "$line")
    seqlen=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['seq_len'])" "$line")
    ref=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['out'])" "$line")
    out_k="${ref/ref__/catk__}"
    run_cat "$prompt" "$seqlen" "$out_k" "-k"

    local sum_k sum_nok=""
    sum_k=$(diff_summary "$ref" "$out_k")

    if [[ -n "${TALKIE_INCLUDE_NOCACHE:-}" ]]; then
      out_nok="${ref/ref__/catn__}"
      run_cat "$prompt" "$seqlen" "$out_nok" ""
      sum_nok=$(diff_summary "$out_k" "$out_nok")
    fi

    # Accumulate token-level stability stats for the headline number.
    if [[ "$sum_k" = "ok" ]]; then
      total_matched=$((total_matched + seqlen))
      total_tokens=$((total_tokens + seqlen))
    else
      local matched=${sum_k#drift }; matched=${matched%%/*}
      local denom=${sum_k#drift *\/}; denom=${denom%% *}
      total_matched=$((total_matched + matched))
      total_tokens=$((total_tokens + denom))
    fi

    local row
    row=$(printf "%-50s s=%-3d  " "$prompt" "$seqlen")
    if [[ "$sum_k" = "ok" && ( -z "$sum_nok" || "$sum_nok" = "ok" ) ]]; then
      printf "%sPASS\n" "$row"
    elif [[ "$sum_k" = "ok" ]]; then
      printf "%sPASS (ref==cat-k); cat-k != cat-nok: %s\n" "$row" "$sum_nok"
      DRIFT=$((DRIFT + 1))
    else
      printf "%sFAIL (ref != cat-k): %s\n" "$row" "$sum_k"
      [[ -n "$sum_nok" && "$sum_nok" != "ok" ]] && printf "        cat-k != cat-nok: %s\n" "$sum_nok"
      FAILED=$((FAILED + 1))
    fi
  done < "$cases_jsonl"

  local pct=0
  [[ "$total_tokens" -gt 0 ]] && pct=$(( total_matched * 1000 / total_tokens ))
  echo
  if [[ -n "${TALKIE_INCLUDE_NOCACHE:-}" ]]; then
    printf "summary: %d cases, %d ref-mismatches, %d cat-k/cat-nok drifts; token match %d/%d (%d.%d%%)\n" \
      "$(wc -l <"$cases_jsonl")" "$FAILED" "$DRIFT" \
      "$total_matched" "$total_tokens" "$((pct / 10))" "$((pct % 10))"
  else
    printf "summary: %d cases, %d ref-mismatches; token match %d/%d (%d.%d%%)\n" \
      "$(wc -l <"$cases_jsonl")" "$FAILED" \
      "$total_matched" "$total_tokens" "$((pct / 10))" "$((pct % 10))"
  fi
  return "$FAILED"
}

# ---------------------------------------------------------------------------
# Single-shot mode
# ---------------------------------------------------------------------------

if [[ -n "${TALKIE_PROMPT:-}" || -n "${TALKIE_SEQLEN:-}" ]]; then
  PROMPTS=("${TALKIE_PROMPT:-Once upon a time}")
  SEQLENS=("${TALKIE_SEQLEN:-40}")
else
  # ---------------------------------------------------------------------------
  # Default matrix: variety of prompts × two lengths.
  # ---------------------------------------------------------------------------
  PROMPTS=(
    "The quick brown fox"
    "Once upon a time"
    "Category theory is"
    "If scientists discover life on other planets,"
    'Mr. Carnegie said: "I am'
  )
  # Talkie's forward has no KV cache, so the Python ref is O(N²).
  # Two lengths balance "do we match at all" against "does drift appear
  # over many tokens" — 40 is enough that any sub-ulp bf16 disagreement
  # on the top two logits will eventually pick a different argmax.
  SEQLENS=(20 40)
fi

FAILED=0
DRIFT=0
run_matrix
