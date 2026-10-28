#!/usr/bin/env bash
# Build vllmv.cu -> ./vllmv
# Usage: ./build.sh [debug]
set -euo pipefail
cd "$(dirname "$0")"

ARCH=${ARCH:-sm_120}   # RTX 5060 (Blackwell, cc 12.0)
FLAGS="-O3"
[[ "${1:-}" == "debug" ]] && FLAGS="-O0 -g -G"

nvcc $FLAGS -std=c++17 -arch=$ARCH \
    -Iinclude \
    -Xcompiler -Wall,-Wextra \
    vllmv.cu -o vllmv -lcublas

echo "built ./vllmv"

#Generated with AI

