# Architecture

## One process

The promoted topology is deliberately simple:

```text
ComfyUI :8188
├── normal image/Krea/custom nodes
├── GLSLShader → Mesa EGL + OpenGL ES 3.2
└── TRELLIS.2 → ROCm native extensions + PyTorch UV texture rasterizer
```

There is no second ComfyUI server, proxy node, or private API port.

ComfyUI normally preloads bundled ANGLE EGL/GLES with global symbol visibility when `nodes_glsl.py` imports. Some TRELLIS render paths use Mesa desktop OpenGL, while the validated texture-bake route uses a ROCm-safe PyTorch rasterizer. The GPL-separated core patch makes ANGLE preload conditional on `COMFYUI_DISABLE_ANGLE_PRELOAD`.

The launcher sets the variable before Python starts, so PyOpenGL resolves Mesa. On the validated RX 7900 XTX process:

- native TRELLIS shape generation completed;
- `GLSLShader` subsequently created `EGL 1.5` and `OpenGL ES 3.2 Mesa` on radeonsi;
- Krea ran before and after TRELLIS;
- only port 8188 was listening.

This preserves the `GLSLShader` node and behavior but changes its implementation backend from ANGLE/Vulkan to Mesa/radeonsi.

## Native model path

`Trellis2LoadModel_ConvRot` passes the combined checkpoint to the native TRELLIS model factory. The model manager still acquires DINO, encoder/decoder payloads, and all 512/1024 architecture JSON files on a clean install, but skips BF16 flow payloads replaced by ConvRot. Checkpoint metadata and available prefixes select the correct 512 or 1024 component. The adapter:

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

`RasterizeCudaContext` is unavailable on the tested ROCm nvdiffrast port. Creating `RasterizeGLContext` late in a loaded 1024 texture job also caused a native segmentation fault, even though standalone Mesa GL rasterization works.

The texture-bake nodes therefore use a bounded triangle-driven PyTorch UV rasterizer on ROCm. It processes faces in chunks, computes pixel-center barycentrics, and returns only covered 3D surface positions. The existing sparse attribute sampler then produces base-color, metallic, roughness, and alpha maps, which are attached as a `PBRMaterial` before GLB export. CUDA builds retain the original nvdiffrast path.

The validated 1024 route generated two 2048² textures and a GLB containing `TEXCOORD_0`, one material, two textures, and two embedded images. Blender 5.1 imported all of them successfully.
