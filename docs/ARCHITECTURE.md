# Architecture

## One process

The promoted topology is deliberately simple:

```text
ComfyUI :8188
├── normal image/Krea/custom nodes
├── GLSLShader → Mesa EGL + OpenGL ES 3.2
└── TRELLIS.2 → Mesa desktop OpenGL + ROCm native extensions
```

There is no second ComfyUI server, proxy node, or private API port.

ComfyUI normally preloads bundled ANGLE EGL/GLES with global symbol visibility when `nodes_glsl.py` imports. TRELLIS's ROCm nvdiffrast plugin needs Mesa desktop OpenGL. The GPL-separated core patch makes that preload conditional on `COMFYUI_DISABLE_ANGLE_PRELOAD`.

The launcher sets the variable before Python starts, so PyOpenGL resolves Mesa. On the validated RX 7900 XTX process:

- native TRELLIS shape generation completed;
- `GLSLShader` subsequently created `EGL 1.5` and `OpenGL ES 3.2 Mesa` on radeonsi;
- Krea ran before and after TRELLIS;
- only port 8188 was listening.

This preserves the `GLSLShader` node and behavior but changes its implementation backend from ANGLE/Vulkan to Mesa/radeonsi.

## Native model path

`Trellis2LoadModel_GGUF` selects `INT8 ConvRot` and passes the combined checkpoint to the native TRELLIS model factory. The model manager still acquires DINO, encoder/decoder payloads, and the three 512 architecture JSON files on a clean install, but skips the three BF16 flow payloads replaced by ConvRot. The adapter:

1. constructs the selected flow model on the meta device;
2. replaces exactly 210 dense/sparse linear modules with `ConvRotLinear`;
3. reads one component's F32, BF16, I8, and U8 runs sequentially;
4. assigns tensors without a generic ComfyUI `ModelPatcher`;
5. materializes RoPE buffers that cannot remain on meta;
6. transfers the owned component model to the GPU.

Cross-attention `to_kv` layers cache the rotated and row-quantized conditioning tensor across blocks and denoising steps. Cache keys include object identity, device, HIP stream, shape, stride, storage offset, dtype, group size, and tensor version where available. Entries are bounded and weak-reference aware.

## Backend license isolation

The MIT TRELLIS adapter loads the patientx backend from `TRELLIS2_INT8_FAST_PATH` under a private Python package name. The backend remains a separate AGPL checkout. Measured TRELLIS static configurations are requested only with `trellis_static=True`; generic backend users retain normal autotuning.

The ComfyUI core, TRELLIS adapter, and backend patches remain separate files with their upstream GPL-3.0, MIT, and AGPL-3.0 notices.

## Rasterization boundary

`RasterizeCudaContext` is not a fallback on the tested ROCm nvdiffrast port; its implementation is an explicit unsupported stub. TRELLIS code must continue using `RasterizeGLContext`.

The completed public validation covers shape generation and export. Although Mesa desktop GL and Mesa GLES coexist in the same process, complete TRELLIS texture generation is not yet promoted as a verified end-to-end route.
