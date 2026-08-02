# TRELLIS.2 INT8 ConvRot for AMD ROCm

Native fused INT8 ConvRot execution for TRELLIS.2 inside a single ComfyUI process on AMD `gfx1100` GPUs.

> **Want the installable custom node instead of a developer patch kit?** Use [`DrBearJew/ComfyUI-Trellis2-ConvRot-ROCM`](https://github.com/DrBearJew/ComfyUI-Trellis2-ConvRot-ROCM) or its [latest release](https://github.com/DrBearJew/ComfyUI-Trellis2-ConvRot-ROCM/releases/latest). It embeds the verified node changes, installs the pinned license-separated backend companion, bundles the standard and fast workflows, and provides a strict ROCm dependency check.

This patch kit adds:

- native `ConvRotLinear` loading for TRELLIS sparse and dense flow models;
- fused W8A8 Triton kernels tuned for measured TRELLIS shapes;
- the published BitPoet checkpoint as a pinned, verified download;
- one-process Mesa support for TRELLIS and ComfyUI `GLSLShader`;
- checkpoint-aware 512, 1024, and 1024-cascade routing;
- ROCm-safe UV rasterization and end-to-end PBR textured GLB export;
- GPU-first CuMesh processing with a bounded ROCm safety preflight;
- safe fallback when Triton is unavailable.

Validated on an RX 7900 XTX with Python 3.12, PyTorch 2.14 ROCm 7.15, and Triton 3.8.

## Latest improvements and bug fixes

- **GPU-first generation:** sampling stages now run with `low_vram=false`, avoiding CPU offload during active generation while still unloading models between major stages with `keep_models_loaded=false`. GPU CuMesh initialization and hole filling replace the former unconditional CPU geometry path.
- **Faster optional UV profile:** the new explicit [fast textured workflow](workflows/trellis2_convrot_bitpoet_1024_textured_fast.workflow.json) uses 20° pre-clustering to reduce CPU Xatlas time. The original 60° workflow remains the quality-first default because the fast profile creates more UV seams.
- **ROCm CuMesh crash fix:** meshes at or above the observed `2^20` vertex-row boundary are pre-simplified before the first CuMesh call, preventing the unrecoverable `hipMemcpy2D: invalid argument` failure while leaving smaller meshes on the direct GPU path. A 1,051,966-vertex / 2,095,944-face validation completed successfully through this preflight.
- **Blender transfer fix:** textured workflows retain CuMesh custom normals, GLB export always emits normalized `NORMAL`, PBR resources receive stable semantic names, and Blender-readable extras identify normalized/non-authoritative scale plus metallic/roughness channel roles. Blender imports custom split normals instead of treating every face as flat. Older GLBs must be re-exported to receive these contracts.
- **RGB/RGBA preprocessing fix:** RGB inputs no longer crash when background removal is disabled; semi-transparent foreground participates in cropping instead of being discarded by an 80% alpha threshold, and fully transparent inputs fail with a clear error.
- **Bounded ROCm UV memory:** the PyTorch UV fallback now chunks pixel candidates as well as faces, preventing a single large UV triangle from allocating an entire high-resolution atlas candidate tensor at once. Direct/retexturing paths share this HIP-safe rasterizer and fail clearly before unsafe CuMesh initialization at the observed vertex-row boundary.
- **No normal-VRAM projection offload:** projected conditioning remains on GPU when `low_vram=false`; CPU concatenation and cache clearing are retained only for explicit low-VRAM mode.
- **Fail-closed projected views:** Pixel3D projected conditioning now rejects zero or multiple views until its batch/grid math is implemented, and image batches above `max_views` fail instead of silently dropping inputs.
- **Reproducible patch coverage:** the published patch now includes the ConvRot loader, ROCm UV rasterizer, GPU-first geometry changes, the CuMesh boundary fix, and normal-preserving GLB export; clean application and the full test suite pass.
- **Experimental native IU4 probe:** [`experiments/gfx1100-iu4`](experiments/gfx1100-iu4) provides an isolated `gfx1100` W4A4 WMMA correctness and assembly harness. It is not enabled in the TRELLIS runtime or presented as a production speed path.

## Results

Observed flow execution times:

| Flow execution | Q4_K_M | INT8 ConvRot | Ratio |
|---|---:|---:|---:|
| Cold structure | 4.927 s | 1.600 s | 3.08× |
| Cold shape | 5.802 s | 2.146 s | 2.70× |
| Warm structure | 0.341 s | 0.243 s | 1.40× |
| Warm shape | 0.791 s | 0.679 s | 1.16× |

A complete 512 shape-only run measured **131.67 s with Q4_K_M** and **104.04 s with INT8 ConvRot**: an observed **1.27× wall-clock improvement**.

The enabled BitPoet **1024 shape** route completed in **127.13 s** and exported a valid 30.47 MB GLB with 819,421 vertices and 1,719,392 faces. The complete **1024 textured** route completed in **213.95 s** and exported a 25.62 MB GLB containing UVs, one PBR material, and two embedded 2048² textures; Blender 5.1 imported the material and both texture images successfully. A later multi-million-face input exposed a ROCm `hipMemcpy2D` failure during CuMesh initialization. Smaller meshes remain GPU-first, while inputs at or above the observed 2^20 vertex-row boundary are pre-simplified on CPU before the first CuMesh call; all CuMesh inputs use contiguous `float32`/`int32` tensors.

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

Three ready-to-use ComfyUI graphs are included:

- [`1024 shape-only`](workflows/trellis2_convrot_bitpoet_1024.workflow.json), with a matching [`API payload`](workflows/trellis2_convrot_bitpoet_1024.api.json);
- [`1024 textured PBR`](workflows/trellis2_convrot_bitpoet_1024_textured.workflow.json), the quality-first 60° UV profile, with a matching [`API payload`](workflows/trellis2_convrot_bitpoet_1024_textured.api.json);
- [`1024 textured PBR — fast UV`](workflows/trellis2_convrot_bitpoet_1024_textured_fast.workflow.json), the explicit 20° profile, with a matching [`API payload`](workflows/trellis2_convrot_bitpoet_1024_textured_fast.api.json).

Download a workflow, drag it onto the canvas, choose an input image, and queue it. All use ComfyUI's standard **Load Image** node and clean ConvRot-specific node names. The fast UV profile reduces CPU Xatlas time but creates more UV seams, so the standard textured workflow remains the quality-first default. Both textured workflows enable custom-normal transfer for Blender-safe GLB shading.

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
- validated 1024 shape-only and 1024 PBR-textured GLB workflows
- one ComfyUI process on port 8188
- Mesa desktop OpenGL for TRELLIS and Mesa GLES for `GLSLShader`
- prebuilt ROCm TRELLIS extensions matching the active Python/PyTorch ABI

## License

Original scripts and documentation are MIT licensed. Patch files retain the licenses of their upstream projects: ComfyUI GPL-3.0, ComfyUI-Trellis2-GGUF MIT, and ComfyUI-INT8-Fast-ROCM AGPL-3.0. Model terms apply separately.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
