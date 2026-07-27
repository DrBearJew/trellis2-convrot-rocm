# TRELLIS.2 INT8 ConvRot for AMD ROCm

Native fused INT8 ConvRot execution for TRELLIS.2 inside a single ComfyUI process on AMD `gfx1100` GPUs.

This patch kit adds:

- native `ConvRotLinear` loading for TRELLIS sparse and dense flow models;
- fused W8A8 Triton kernels tuned for measured TRELLIS shapes;
- the published BitPoet checkpoint as a pinned, verified download;
- one-process Mesa support for TRELLIS and ComfyUI `GLSLShader`;
- checkpoint-aware 512, 1024, and 1024-cascade routing;
- safe fallback when Triton is unavailable.

Validated on an RX 7900 XTX with Python 3.12, PyTorch 2.14 ROCm 7.15, and Triton 3.8.

## Results

Observed flow execution times:

| Flow execution | Q4_K_M | INT8 ConvRot | Ratio |
|---|---:|---:|---:|
| Cold structure | 4.927 s | 1.600 s | 3.08× |
| Cold shape | 5.802 s | 2.146 s | 2.70× |
| Warm structure | 0.341 s | 0.243 s | 1.40× |
| Warm shape | 0.791 s | 0.679 s | 1.16× |

A complete 512 shape-only run measured **131.67 s with Q4_K_M** and **104.04 s with INT8 ConvRot**: an observed **1.27× wall-clock improvement**.

The enabled BitPoet **1024** route completed in **127.13 s** and exported a valid 30.47 MB GLB with 819,421 vertices and 1,719,392 faces.

INT8 is optimized for execution speed, not model size. The three-component rebuilt INT8 checkpoint occupies 3.94 GB versus 2.37 GB for the corresponding Q4_K_M files, about 1.66× larger. The two measured runs also produced different mesh topology, so compare visual output for your workload rather than treating polygon count as a quality score.

Full measurements: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

## Pinned components

| Component | Revision |
|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `c2638ce6c00e3426c48d56a775bc46e9a8464094` |
| [ComfyUI-Trellis2-GGUF](https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF) | `6bd11ead7ab7976ec4b2c47db52701f4c76a54e2` |
| [ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM) | `5e365a2d02058a3c6d57405ae07bb99a3804c7cc` |
| [BitPoet/TRELLIS.2-int8-convrot](https://huggingface.co/BitPoet/TRELLIS.2-int8-convrot) | `2f7cd18627fc89c9f238e63bdd0abb5b204d13c1` |

## Quick start

Set paths for your installation:

```bash
export KIT="$PWD"
export COMFYUI="$HOME/apps/ComfyUI"
export TRELLIS_NODE="$COMFYUI/custom_nodes/ComfyUI-Trellis2-GGUF"
export INT8_BACKEND="$HOME/src/ComfyUI-INT8-Fast-ROCM"
export TRELLIS_PYTHON="$HOME/venvs/trellis/bin/python"
export ROCM_PYTHON_ENV="$HOME/venvs/rocm-runtime"
```

Clone the pinned repositories as described in [docs/INSTALL.md](docs/INSTALL.md), then apply the patches:

```bash
./scripts/apply-patches.sh \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"
```

Download and verify the ready-to-use BitPoet checkpoint:

```bash
"$TRELLIS_PYTHON" scripts/verify-bitpoet-checkpoint.py \
  --download-to "$COMFYUI/models/diffusion_models"
```

Start ComfyUI:

```bash
./scripts/start-comfyui-native.sh \
  --comfyui "$COMFYUI" \
  --python "$TRELLIS_PYTHON" \
  --int8-backend "$INT8_BACKEND" \
  --rocm-python-env "$ROCM_PYTHON_ENV" \
  --listen 127.0.0.1 \
  --port 8188
```

Use the dedicated **Trellis2 - Load Model (INT8 ConvRot)** node. The BitPoet checkpoint supports 512 shape generation plus complete 1024 and 1024-cascade flow routing. The locally rebuilt three-component checkpoint supports the 512 route.

> **Naming note:** `ComfyUI-Trellis2-GGUF` is the upstream extension name. Existing `_GGUF` node IDs remain available only for compatibility, while this patch adds format-neutral aliases and a dedicated ConvRot loader. The ConvRot route loads the `.safetensors` checkpoint above; it does **not** require TRELLIS GGUF flow weights.

For a ready-to-use ComfyUI graph, download and drag [`workflows/trellis2_convrot_bitpoet_1024.workflow.json`](workflows/trellis2_convrot_bitpoet_1024.workflow.json) onto the canvas, choose an input image, and queue it. The graph uses ComfyUI's standard **Load Image** node and explicit INT8 ConvRot titles. A separate [`API payload`](workflows/trellis2_convrot_bitpoet_1024.api.json) is included for automation.

The first load downloads the normal TRELLIS support assets such as DINO, encoders, decoders, and architecture configs. It does not download duplicate BF16 flow weights.

## Build the checkpoint yourself

To derive a smaller three-component checkpoint directly from the pinned Microsoft BF16 sources:

```bash
"$TRELLIS_PYTHON" scripts/build-checkpoint.py \
  --output "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors" \
  --int8-backend "$INT8_BACKEND" \
  --device cuda

"$TRELLIS_PYTHON" scripts/validate-checkpoint.py \
  "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors"
```

See [docs/CHECKPOINT.md](docs/CHECKPOINT.md) for the checkpoint format.

## Scope

- RX 7900 XTX / `gfx1100`
- 512, 1024, and 1024-cascade TRELLIS routing with the BitPoet checkpoint
- one ComfyUI process on port 8188
- Mesa desktop OpenGL for TRELLIS and Mesa GLES for `GLSLShader`
- prebuilt ROCm TRELLIS extensions matching the active Python/PyTorch ABI

## License

Original scripts and documentation are MIT licensed. Patch files retain the licenses of their upstream projects: ComfyUI GPL-3.0, ComfyUI-Trellis2-GGUF MIT, and ComfyUI-INT8-Fast-ROCM AGPL-3.0. Model terms apply separately.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
