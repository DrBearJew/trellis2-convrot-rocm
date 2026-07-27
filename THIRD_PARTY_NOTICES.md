# Third-party notices

This repository is a patch kit. Its root MIT license covers original scripts and documentation only; it does not relicense upstream code, patch hunks, or models.

## ComfyUI

- Upstream: <https://github.com/comfyanonymous/ComfyUI>
- Pinned revision: `c2638ce6c00e3426c48d56a775bc46e9a8464094`
- License: GNU General Public License v3.0
- Patch: `patches/comfyui-core-gpl3.patch`
- License copy: `LICENSES/ComfyUI-GPL-3.0.txt`

## ComfyUI-Trellis2-GGUF

- Upstream: <https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF>
- Pinned revision: `6bd11ead7ab7976ec4b2c47db52701f4c76a54e2`
- License: MIT
- Patch: `patches/trellis2-gguf-mit.patch`
- License copy: `LICENSES/ComfyUI-Trellis2-GGUF-MIT.txt`

## ComfyUI-INT8-Fast-ROCM

- Upstream: <https://github.com/patientx/ComfyUI-INT8-Fast-ROCM>
- Pinned revision: `5e365a2d02058a3c6d57405ae07bb99a3804c7cc`
- License: GNU Affero General Public License v3.0
- Patch: `patches/int8-fast-rocm-agpl.patch`
- License copy: `LICENSES/ComfyUI-INT8-Fast-ROCM-AGPL-3.0.txt`

The backend remains an external dependency loaded from `TRELLIS2_INT8_FAST_PATH`; AGPL backend files are not embedded inside the MIT TRELLIS custom node.

## BitPoet community checkpoint

- Artifact repository: <https://huggingface.co/BitPoet/TRELLIS.2-int8-convrot>
- Pinned revision: `2f7cd18627fc89c9f238e63bdd0abb5b204d13c1`
- Checkpoint SHA256: `66d269c1f874d38fe491a413e16944ff208a4ae348e01fc3e97b5531b52a7f3f`

The artifact is not redistributed here. Its model card and the underlying TRELLIS.2 source terms govern use; verify those terms before redistribution.

## TRELLIS.2 model sources

The checkpoint builder downloads three source files from:

- Model repository: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- Pinned revision: `af44b45f2e35a493886929c6d786e563ec68364d`

No model weights are included. Source and derived checkpoints remain subject to their model licenses and hosting terms.
