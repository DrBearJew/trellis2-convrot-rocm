#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  apply-patches.sh \
    --comfyui PATH \
    --trellis-node PATH \
    --int8-backend PATH \
    [--reverse]

Applies the three license-separated patches to clean repositories at the pinned
commits documented in README.md. --reverse checks every patch before changing
any repository.
EOF
}

COMFYUI=""
TRELLIS_NODE=""
INT8_BACKEND=""
REVERSE=0
while (($#)); do
  case "$1" in
    --comfyui) COMFYUI=${2:?}; shift 2 ;;
    --trellis-node) TRELLIS_NODE=${2:?}; shift 2 ;;
    --int8-backend) INT8_BACKEND=${2:?}; shift 2 ;;
    --reverse) REVERSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$COMFYUI" && -n "$TRELLIS_NODE" && -n "$INT8_BACKEND" ]] || {
  usage >&2
  exit 2
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMFYUI_COMMIT=c2638ce6c00e3426c48d56a775bc46e9a8464094
TRELLIS_COMMIT=6bd11ead7ab7976ec4b2c47db52701f4c76a54e2
BACKEND_COMMIT=5e365a2d02058a3c6d57405ae07bb99a3804c7cc
CORE_PATCH="$ROOT/patches/comfyui-core-gpl3.patch"
TRELLIS_PATCH="$ROOT/patches/trellis2-gguf-mit.patch"
BACKEND_PATCH="$ROOT/patches/int8-fast-rocm-agpl.patch"

check_repo() {
  local path=$1 expected=$2 label=$3
  git -C "$path" rev-parse --is-inside-work-tree >/dev/null
  local actual
  actual=$(git -C "$path" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] || {
    echo "$label must be at $expected, found $actual" >&2
    exit 1
  }
  [[ -z $(git -C "$path" status --porcelain) ]] || {
    echo "$label has local changes; refusing to patch: $path" >&2
    exit 1
  }
}

if ((REVERSE)); then
  git -C "$COMFYUI" apply --check --reverse "$CORE_PATCH"
  git -C "$TRELLIS_NODE" apply --check --reverse "$TRELLIS_PATCH"
  git -C "$INT8_BACKEND" apply --check --reverse "$BACKEND_PATCH"
  git -C "$COMFYUI" apply --reverse "$CORE_PATCH"
  git -C "$TRELLIS_NODE" apply --reverse "$TRELLIS_PATCH"
  git -C "$INT8_BACKEND" apply --reverse "$BACKEND_PATCH"
  echo "All three patches reversed. No model files were changed."
else
  check_repo "$COMFYUI" "$COMFYUI_COMMIT" "ComfyUI"
  check_repo "$TRELLIS_NODE" "$TRELLIS_COMMIT" "ComfyUI-Trellis2-GGUF"
  check_repo "$INT8_BACKEND" "$BACKEND_COMMIT" "ComfyUI-INT8-Fast-ROCM"
  git -C "$COMFYUI" apply --check "$CORE_PATCH"
  git -C "$TRELLIS_NODE" apply --check "$TRELLIS_PATCH"
  git -C "$INT8_BACKEND" apply --check "$BACKEND_PATCH"
  git -C "$COMFYUI" apply "$CORE_PATCH"
  git -C "$TRELLIS_NODE" apply "$TRELLIS_PATCH"
  git -C "$INT8_BACKEND" apply "$BACKEND_PATCH"
  echo "All three patches applied."
fi
