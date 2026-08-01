#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
LLVM="${LLVM:-/opt/rocm/lib/llvm/bin}"
OBJDUMP="${OBJDUMP:-$LLVM/llvm-objdump}"
BUNDLER="${BUNDLER:-$LLVM/clang-offload-bundler}"
GPU="${GPU:-gfx1100}"
mkdir -p "$BUILD"

COMMON=(-O3 -std=c++17 --offload-arch="$GPU")
"$HIPCC" "${COMMON[@]}" "$ROOT/iu4_gemm_probe.hip" -o "$BUILD/iu4_gemm_probe"
"$HIPCC" "${COMMON[@]}" --genco "$ROOT/iu4_gemm_probe.hip" -o "$BUILD/iu4_gemm_probe.bundle"
"$BUNDLER" --unbundle --type=o --input="$BUILD/iu4_gemm_probe.bundle" \
    --targets="hipv4-amdgcn-amd-amdhsa--$GPU" --output="$BUILD/iu4_gemm_probe.hsaco"
"$OBJDUMP" --disassemble --mcpu="$GPU" "$BUILD/iu4_gemm_probe.hsaco" > "$BUILD/disassembly.txt"

IU4_COUNT="$(grep -c 'v_wmma_i32_16x16x16_iu4' "$BUILD/disassembly.txt" || true)"
IU8_COUNT="$(grep -c 'v_wmma_i32_16x16x16_iu8' "$BUILD/disassembly.txt" || true)"
DOT4_COUNT="$(grep -Ec 'v_dot4_i32_(i8|iu8)' "$BUILD/disassembly.txt" || true)"
printf 'assembly iu4=%s iu8_wmma=%s dot4_i8=%s\n' "$IU4_COUNT" "$IU8_COUNT" "$DOT4_COUNT"

if [[ "$IU4_COUNT" -lt 1 || "$IU8_COUNT" -lt 1 || "$DOT4_COUNT" -lt 1 ]]; then
    echo "required gfx1100 integer instruction missing from generated assembly" >&2
    exit 1
fi

"$BUILD/iu4_gemm_probe" --correctness
