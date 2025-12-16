#!/usr/bin/env bash
#
# # Create a shared object from a MLIR text input
#

set -euo pipefail

output_so="$1"

# Default tool names that can be overridden via environment variables
mlir_opt="${MLIR_OPT:-mlir-opt}"
mlir_translate="${MLIR_TRANSLATE:-mlir-translate}"
llc="${LLC:-llc}"
clang="${CLANG:-clang}"

# Temporary files
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/catgrad-mlir.XXXXXXXX")"
cleanup() {
  rm -rf "${tmp_dir}"
}

trap cleanup EXIT

base_name="$(basename "${output_so}")"
base_no_ext="${base_name%.*}"
temp_mlir_in="${tmp_dir}/${base_no_ext}_in.mlir"
temp_mlir_out="${tmp_dir}/${base_no_ext}_out.mlir"
temp_ll="${tmp_dir}/${base_no_ext}.ll"
temp_obj="${tmp_dir}/${base_no_ext}.o"


# Save input MLIR
cat > "${temp_mlir_in}"

mlir_opt_args=(
  --mlir-print-debuginfo
  --convert-elementwise-to-linalg
  --linalg-fuse-elementwise-ops
  --one-shot-bufferize=bufferize-function-boundaries
  --convert-linalg-to-loops
  --convert-scf-to-cf
  --expand-strided-metadata
  --lower-affine
  --finalize-memref-to-llvm
  --convert-math-to-llvm
  --convert-arith-to-llvm
  --convert-func-to-llvm
  --convert-cf-to-llvm
  --ensure-debug-info-scope-on-llvm-func
  --reconcile-unrealized-casts
  --remove-dead-values
)

"${mlir_opt}" "${mlir_opt_args[@]}" "${temp_mlir_in}" > "${temp_mlir_out}"

"${mlir_translate}" --mlir-to-llvmir "${temp_mlir_out}"  >"${temp_ll}"

"${llc}" -filetype=obj -relocation-model=pic "${temp_ll}" -o "${temp_obj}"

"${clang}" -shared -fPIC "${temp_obj}" -o "${output_so}" -lm -lmlir_c_runner_utils
