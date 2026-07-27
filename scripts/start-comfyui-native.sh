#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start-comfyui-native.sh \
    --comfyui PATH \
    --python PATH \
    --int8-backend PATH \
    --rocm-python-env PATH \
    [--listen ADDRESS] [--port PORT] [-- COMFYUI_ARGS...]

Runs one ComfyUI process with native TRELLIS.2 and Mesa EGL. The Python
interpreter must already contain the ROCm TRELLIS geometry extensions.
EOF
}

COMFYUI=""
PYTHON_BIN=""
INT8_BACKEND=""
ROCM_PYTHON_ENV=""
LISTEN="127.0.0.1"
PORT="8188"
EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --comfyui) COMFYUI=${2:?}; shift 2 ;;
    --python) PYTHON_BIN=${2:?}; shift 2 ;;
    --int8-backend) INT8_BACKEND=${2:?}; shift 2 ;;
    --rocm-python-env) ROCM_PYTHON_ENV=${2:?}; shift 2 ;;
    --listen) LISTEN=${2:?}; shift 2 ;;
    --port) PORT=${2:?}; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in COMFYUI PYTHON_BIN INT8_BACKEND ROCM_PYTHON_ENV; do
  [[ -n ${!value} ]] || { echo "Missing required option for $value" >&2; usage >&2; exit 2; }
done
COMFYUI=$(realpath "$COMFYUI")
PYTHON_BIN=$(realpath "$PYTHON_BIN")
INT8_BACKEND=$(realpath "$INT8_BACKEND")
ROCM_PYTHON_ENV=$(realpath "$ROCM_PYTHON_ENV")
[[ -f "$COMFYUI/main.py" ]] || { echo "ComfyUI main.py not found below $COMFYUI" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 1; }
[[ -f "$INT8_BACKEND/int8_fused_kernel.py" && -f "$INT8_BACKEND/convrot.py" ]] || {
  echo "Patched INT8 backend not found: $INT8_BACKEND" >&2; exit 1;
}
[[ -d "$COMFYUI/custom_nodes/ComfyUI-Trellis2-GGUF" ]] || {
  echo "TRELLIS custom node not found below $COMFYUI/custom_nodes" >&2; exit 1;
}

PYTHON_TAG=$($PYTHON_BIN -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
ROCM_DEVEL="$ROCM_PYTHON_ENV/lib/$PYTHON_TAG/site-packages/_rocm_sdk_devel"
ROCM_CORE="$ROCM_PYTHON_ENV/lib/$PYTHON_TAG/site-packages/_rocm_sdk_core"
[[ -d "$ROCM_DEVEL/lib" && -d "$ROCM_CORE/lib" ]] || {
  echo "ROCm SDK packages were not found under $ROCM_PYTHON_ENV/lib/$PYTHON_TAG/site-packages" >&2
  exit 1
}

export ROCM_HOME="$ROCM_DEVEL"
export HIP_HOME="$ROCM_DEVEL"
export CUDA_HOME="$ROCM_DEVEL"
export LD_LIBRARY_PATH="$ROCM_DEVEL/lib:$ROCM_DEVEL/lib/host-math/lib:$ROCM_CORE/lib:$ROCM_CORE/lib/host-math/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$ROCM_PYTHON_ENV/lib/libcrypto.so.3" ]]; then
  export LD_PRELOAD="$ROCM_PYTHON_ENV/lib/libcrypto.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-$PYTORCH_ROCM_ARCH}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-$PYTORCH_ROCM_ARCH}"
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export TRELLIS2_MODEL_RESOLUTION=512
export TRELLIS2_INT8_FAST_PATH="$INT8_BACKEND"
export TRELLIS2_CONVROT_CHECKPOINT="${TRELLIS2_CONVROT_CHECKPOINT:-$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors}"
export COMFYUI_DISABLE_ANGLE_PRELOAD=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/trellis2-convrot/triton}"
export TMPDIR="${TMPDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/trellis2-convrot/tmp}"
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.version.hip:
    raise SystemExit("A working PyTorch ROCm runtime is required")
print(f"PyTorch {torch.__version__} | ROCm {torch.version.hip} | {torch.cuda.get_device_name(0)}")
PY

cd "$COMFYUI"
exec "$PYTHON_BIN" -u main.py \
  --listen "$LISTEN" \
  --port "$PORT" \
  --reserve-vram "${COMFYUI_RESERVE_VRAM:-1}" \
  --async-offload \
  --disable-mmap \
  "${EXTRA_ARGS[@]}"
