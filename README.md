# TRELLIS.2 ConvRot on ROCm (`gfx1100`)

A reviewable patch kit for native TRELLIS.2 INT8 ConvRot execution inside **one ComfyUI process** on AMD ROCm. Mesa supplies both TRELLIS desktop OpenGL and the core `GLSLShader` OpenGL ES context; no second ComfyUI server or proxy API is required.

> **Validated scope:** RX 7900 XTX (`gfx1100`), Python 3.12, PyTorch 2.14 ROCm 7.15 development build, Triton 3.8, ComfyUI at the pinned revision below, and the 512 pipeline. Native ROCm geometry extensions must already be built for the same Python/PyTorch ABI. No model weights, generated assets, credentials, or machine-specific paths are included.

## What this kit contains

- Native `ConvRotLinear` construction for TRELLIS sparse and dense flow models.
- Verifier for the published BitPoet checkpoint plus an exact-manifest three-component builder/validator.
- Prequantized W8A8 Triton execution with opt-in measured `gfx1100` configurations.
- Bounded, inference-safe conditioning preprocessing cache.
- Sequential component loading for slow NTFS/FUSE-hosted checkpoints.
- Correct eager fallback when Triton is unavailable.
- Early 512-only guards on public, split, cascade, refine, and texture routes.
- A small ComfyUI core patch that makes ANGLE preload optional.
- A one-process Mesa launcher contract.

## Pinned revisions and licenses

| Component | Revision | Patch license boundary |
|---|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `c2638ce6c00e3426c48d56a775bc46e9a8464094` | GPL-3.0 |
| [Aero-Ex/ComfyUI-Trellis2-GGUF](https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF) | `6bd11ead7ab7976ec4b2c47db52701f4c76a54e2` | MIT |
| [patientx/ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM) | `5e365a2d02058a3c6d57405ae07bb99a3804c7cc` | AGPL-3.0 |
| [BitPoet/TRELLIS.2-int8-convrot](https://huggingface.co/BitPoet/TRELLIS.2-int8-convrot) ready checkpoint | `2f7cd18627fc89c9f238e63bdd0abb5b204d13c1` | community artifact; source model terms apply |
| [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) source weights | `af44b45f2e35a493886929c6d786e563ec68364d` | model terms apply |

The root MIT license covers original glue scripts and documentation only. It does not relicense patch hunks or models. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Quick start

Define every path explicitly:

```bash
export KIT="$PWD"
export COMFYUI="$HOME/apps/ComfyUI"
export TRELLIS_NODE="$COMFYUI/custom_nodes/ComfyUI-Trellis2-GGUF"
export INT8_BACKEND="$HOME/src/ComfyUI-INT8-Fast-ROCM"
export TRELLIS_PYTHON="$HOME/venvs/trellis/bin/python"
export ROCM_PYTHON_ENV="$HOME/venvs/rocm-runtime"
```

Clone the three pinned repositories, then apply all license-separated patches:

```bash
./scripts/apply-patches.sh \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"
```

Choose one checkpoint path.

**Fast path: download the published BitPoet artifact** (5,253,048,192 bytes; four components, of which the 512 runtime consumes three):

```bash
"$TRELLIS_PYTHON" scripts/verify-bitpoet-checkpoint.py \
  --download-to "$COMFYUI/models/diffusion_models"
```

The helper pins Hugging Face revision `2f7cd18627fc89c9f238e63bdd0abb5b204d13c1`, then verifies the 5.25 GB file's SHA256 and exact runtime tensor schemas.

**Reproducible-build path:** derive a smaller exact three-component checkpoint from the pinned official BF16 files:

```bash
"$TRELLIS_PYTHON" scripts/build-checkpoint.py \
  --output "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors" \
  --int8-backend "$INT8_BACKEND" \
  --device cuda

"$TRELLIS_PYTHON" scripts/validate-checkpoint.py \
  "$COMFYUI/models/diffusion_models/trellis_2_int8_convrot.safetensors"
```

Start one ComfyUI server:

```bash
./scripts/start-comfyui-native.sh \
  --comfyui "$COMFYUI" \
  --python "$TRELLIS_PYTHON" \
  --int8-backend "$INT8_BACKEND" \
  --rocm-python-env "$ROCM_PYTHON_ENV" \
  --listen 127.0.0.1 \
  --port 8188
```

Select **INT8 ConvRot** in `Trellis2LoadModel_GGUF` and use `pipeline_type=512`. On a clean model root, the first load acquires DINO, encoder/decoder assets, and the three flow architecture JSON files. It deliberately does not download the replaced BF16 flow weights.

See [docs/INSTALL.md](docs/INSTALL.md) for prerequisites, verification, and rollback.

## Measured observations

Single-run flow execution observations on the validated machine:

| Flow execution | Q4_K_M | INT8 ConvRot | Observed ratio |
|---|---:|---:|---:|
| Cold structure | 4.927 s | 1.600 s | 3.08× |
| Cold shape | 5.802 s | 2.146 s | 2.70× |
| Warm structure | 0.341 s | 0.243 s | 1.40× |
| Warm shape | 0.791 s | 0.679 s | 1.16× |

These are profiling observations, not statistical benchmark results. The recorded end-to-end runs produced materially different mesh complexity, so this repository makes **no end-to-end speedup claim**. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Important limitations

- **512 only.** Every 1024/cascade route fails early.
- **Three runtime flow components.** The runtime consumes structure, 512 image-to-shape, and 512 shape-to-texture. The pinned BitPoet artifact also contains one unused 1024 image-to-shape component and is verified by exact file hash; locally rebuilt v1 files contain exactly the three runtime components and use the strict manifest validator.
- **Shape-only is the completed end-to-end validation.** Mesa `GLSLShader` and native TRELLIS shape generation coexist in one process; complete textured output remains outside the promoted validation claim.
- **No model redistribution.** The builder downloads pinned source files directly from Hugging Face and writes the derived checkpoint locally.
- **Advanced ROCm install.** This kit does not compile CuMesh, FlexGEMM, O-Voxel, or nvdiffrast. Use an interpreter where those extensions already import.
- Static Triton configurations are opt-in only from native TRELLIS calls; generic INT8/Krea calls retain normal autotuning.
- On low-RAM systems, restarting ComfyUI can be safer than `/free` after loading a multi-gigabyte model because offloading may pressure swap.

## Repository layout

```text
patches/    GPL, MIT, and AGPL upstream patches kept separate
scripts/    apply/reverse, checkpoint build/validation, launcher, checks
manifests/  exact pinned tensor/dtype/shape and provenance contract
tests/      sparse exact-manifest and clean-root acquisition regressions
docs/       installation, checkpoint, architecture, benchmark notes
LICENSES/   exact upstream license copies
```

## Rollback

Stop ComfyUI, then reverse all three patches atomically:

```bash
./scripts/apply-patches.sh --reverse \
  --comfyui "$COMFYUI" \
  --trellis-node "$TRELLIS_NODE" \
  --int8-backend "$INT8_BACKEND"
```

No model files are deleted.
