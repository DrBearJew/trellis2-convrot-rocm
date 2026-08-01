# Installation

## 1. Prerequisites

This is a runtime patch kit, not a general ROCm/TRELLIS installer. Before applying it, one Python 3.12 interpreter must successfully import:

- a ROCm-enabled PyTorch build;
- Triton;
- CuMesh, FlexGEMM, O-Voxel, nvdiffrast, and nvdiffrec;
- the normal `ComfyUI-Trellis2-GGUF` Python dependencies.

All compiled extensions must use the same Python/PyTorch/ROCm ABI used to launch ComfyUI.

Validated stack:

- RX 7900 XTX (`gfx1100`)
- Python `3.12.12`
- PyTorch `2.14.0a0+rocm7.15.0a20260712`
- ROCm `7.15.0`
- Triton `3.8.0`

Both the shape-only and complete PBR-textured 1024 workflows have been exercised on the validated stack. On ROCm, the texture bake uses the included bounded PyTorch UV rasterizer instead of nvdiffrast OpenGL interop.

## 2. Define paths

Run these from the patch-kit checkout. Change the values for your machine; do not leave variables undefined.

```bash
export KIT="$(pwd)"
export COMFYUI="$HOME/apps/ComfyUI"
export TRELLIS_NODE="$COMFYUI/custom_nodes/ComfyUI-Trellis2-GGUF"
export INT8_BACKEND="$HOME/src/ComfyUI-INT8-Fast-ROCM"
export TRELLIS_PYTHON="$HOME/venvs/trellis/bin/python"
export ROCM_PYTHON_ENV="$HOME/venvs/rocm-runtime"
```

`ROCM_PYTHON_ENV` is the Python environment containing `_rocm_sdk_devel` and `_rocm_sdk_core`. `TRELLIS_PYTHON` is the interpreter containing the native TRELLIS extensions. They may refer to the same environment.

Check every value before mutation:

```bash
printf '%s\n' "$KIT" "$COMFYUI" "$TRELLIS_NODE" "$INT8_BACKEND" "$TRELLIS_PYTHON" "$ROCM_PYTHON_ENV"
test -x "$TRELLIS_PYTHON"
```

## 3. Clone pinned sources

If ComfyUI is not already present:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI"
git -C "$COMFYUI" checkout c2638ce6c00e3426c48d56a775bc46e9a8464094
```

Clone the custom node directly into this ComfyUI and keep the AGPL backend external:

```bash
git clone https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF.git "$TRELLIS_NODE"
git -C "$TRELLIS_NODE" checkout 6bd11ead7ab7976ec4b2c47db52701f4c76a54e2

mkdir -p "$(dirname "$INT8_BACKEND")"
git clone https://github.com/patientx/ComfyUI-INT8-Fast-ROCM.git "$INT8_BACKEND"
git -C "$INT8_BACKEND" checkout 5e365a2d02058a3c6d57405ae07bb99a3804c7cc
```

The patch command refuses mismatched revisions or dirty repositories.

## 4. Apply all three patches

```bash
"$KIT/scripts/apply-patches.sh" \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"
```

The patches remain license-separated:

- `patches/comfyui-core-gpl3.patch`: conditional ANGLE preload;
- `patches/trellis2-gguf-mit.patch`: native TRELLIS adapter;
- `patches/int8-fast-rocm-agpl.patch`: fused backend/cache support.

## 5. Obtain the checkpoint

### 5A. Download the published BitPoet artifact

The runtime was validated with `BitPoet/TRELLIS.2-int8-convrot` revision `2f7cd18627fc89c9f238e63bdd0abb5b204d13c1`. Pin the revision and verify the exact 5.25 GB artifact before use:

```bash
mkdir -p "$COMFYUI/models/diffusion_models"
"$TRELLIS_PYTHON" "$KIT/scripts/verify-bitpoet-checkpoint.py" \
  --download-to "$COMFYUI/models/diffusion_models"
```

The helper uses `huggingface_hub` to pin the repository revision; no floating `main` download is accepted.

The verifier requires SHA256 `66d269c1f874d38fe491a413e16944ff208a4ae348e01fc3e97b5531b52a7f3f` and confirms all 4,240 tensors. Runtime routing uses `img2shape_512` for 512 shape generation, `img2shape` for 1024 shape generation, and `shape2txt` for 1024 texturing.

### 5B. Rebuild from pinned official BF16 sources

For an independently reproducible and smaller three-component artifact, the builder downloads exactly three BF16 flow files from `microsoft/TRELLIS.2-4B` revision `af44b45f2e35a493886929c6d786e563ec68364d`, rotates 210 eligible linear weights per component, quantizes them row-wise to INT8, and streams a combined safetensors file.

Expect roughly 8 GB of source downloads, a roughly 4 GB result, temporary disk use, and approximately 1.5 GB of working RAM plus the selected GPU.

```bash
mkdir -p "$COMFYUI/models/diffusion_models"
"$TRELLIS_PYTHON" "$KIT/scripts/build-checkpoint.py" \
  --output "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors" \
  --int8-backend "$INT8_BACKEND" \
  --device cuda
```

For previously downloaded sources, pass all three overrides together:

```bash
"$TRELLIS_PYTHON" "$KIT/scripts/build-checkpoint.py" \
  --output "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors" \
  --int8-backend "$INT8_BACKEND" \
  --structure /path/to/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  --shape /path/to/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --texture /path/to/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
  --device cuda
```

The builder refuses to overwrite an existing output. It also verifies the backend Git revision and embeds the pinned source/backend provenance required by `manifests/trellis2-convrot-v1.json`.

## 6. Validate the checkpoint and runtime contract

Export the selected artifact. Path 5A is validated by `verify-bitpoet-checkpoint.py`; path 5B is validated by the exact v1 manifest:

```bash
export TRELLIS2_CONVROT_CHECKPOINT="$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors"

# Reproducible-build path 5B only:
"$TRELLIS_PYTHON" "$KIT/scripts/validate-checkpoint.py" \
  "$TRELLIS2_CONVROT_CHECKPOINT"

# Both paths:
"$TRELLIS_PYTHON" "$KIT/scripts/verify-native-contract.py" \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"

cd "$INT8_BACKEND"
"$TRELLIS_PYTHON" -m pytest -q \
  tests/test_int8_prequantized.py \
  tests/test_convrot_activation_cache.py
cd "$KIT"
```

## 7. Start one ComfyUI process

The launcher exports `COMFYUI_DISABLE_ANGLE_PRELOAD=1`. This leaves Mesa EGL authoritative for nvdiffrast. The existing core `GLSLShader` node then obtains a Mesa OpenGL ES context instead of ANGLE.

```bash
"$KIT/scripts/start-comfyui-native.sh" \
  --comfyui "$COMFYUI" \
  --python "$TRELLIS_PYTHON" \
  --int8-backend "$INT8_BACKEND" \
  --rocm-python-env "$ROCM_PYTHON_ENV" \
  --listen 127.0.0.1 \
  --port 8188
```

There is no backend port and no proxy node. To listen beyond loopback, add your own authentication and network controls.

In the UI:

1. confirm `GLSL Shader` exists;
2. use the dedicated **Trellis2 - Load Model (INT8 ConvRot)** node;
3. use `pipeline_type=1024` or `1024_cascade` with the BitPoet checkpoint;
4. use `pipeline_type=512` for 512 shape generation or with the locally rebuilt v1 checkpoint;
5. for textured output, keep `generate_texture_slat=true` and connect both the mesh and BVH outputs to **Trellis2 - Postprocess and Texture Bake** before exporting the GLB.

On ROCm, the postprocess node keeps smaller meshes GPU-first but pre-simplifies inputs at or above the observed 2^20 vertex-row boundary with Meshlib before the first CuMesh call. This prevents an unrecoverable-in-process `hipMemcpy2D: invalid argument` while preserving the fast path for the measured 866,682-vertex / 1,728,856-face input. The normal remesh and simplify stages still apply afterward, and CuMesh inputs are normalized to contiguous `float32` vertices and `int32` faces. CUDA retains its original no-pre-simplification path.

The upstream extension directory retains `GGUF`, and its original `_GGUF` class IDs remain registered for compatibility with existing workflows. This patch adds clean aliases and a dedicated ConvRot loader for new workflows. The ConvRot route loads `trellis_2_int8_convrot.safetensors` and does not require TRELLIS GGUF flow weights. The included GUI workflows also use ComfyUI's standard **Load Image** node.

On a clean `models/Trellis2` root, that first model selection acquires the normal non-flow runtime assets: `pipeline.json`, DINO, encoders/decoders, and all 512/1024 architecture JSON files. ConvRot replaces the BF16 flow payloads, so those multi-gigabyte weights are not downloaded a second time. Background-removal assets retain the upstream node's normal lazy acquisition behavior.

## Rollback

Stop the launcher. Confirm the same variables from section 2 are still defined, then run:

```bash
: "${COMFYUI:?COMFYUI is required}"
: "${TRELLIS_NODE:?TRELLIS_NODE is required}"
: "${INT8_BACKEND:?INT8_BACKEND is required}"

"$KIT/scripts/apply-patches.sh" --reverse \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"
```

Rollback reverses only the three patches. It does not remove repositories, environments, caches, or models.
