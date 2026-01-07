#!/usr/bin/env bash
set -euxo

# compile helper
gcc -o main main.c

../scripts/llvm.sh main.mlir > lowered.mlir
mlir-translate lowered.mlir --mlir-to-llvmir -o main.ll
clang -shared -Wno-override-module main.ll -o main.so -lm
