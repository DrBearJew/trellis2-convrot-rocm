# Checkpoint contract

## Supported runtime components

The locally rebuilt v1 checkpoint contains the three 512 components:

```text
model.structure_model.*
model.img2shape_512.*
model.shape2txt.*
```

For this metadata-tagged v1 layout, `shape2txt` is the 512 texture flow. They map to the pinned BF16 source files:

```text
ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors
```

The builder pins `microsoft/TRELLIS.2-4B` revision `af44b45f2e35a493886929c6d786e563ec68364d`. `manifests/trellis2-convrot-v1.json` records every expected output tensor name, dtype, and shape plus all three source SHA256 values and the pinned backend revision.

## Per-component schema

Each source component has 640 BF16 tensors. Exactly 210 eligible two-dimensional linear weights are converted to:

```text
<module>.weight         I8  [out_features, in_features]
<module>.weight_scale   F32 [out_features, 1]
<module>.comfy_quant    U8  JSON bytes
```

Each `<module>.comfy_quant` tensor contains this JSON:

```json
{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

The five intentionally unquantized two-dimensional modules are:

```text
adaLN_modulation.1
input_layer
out_layer
t_embedder.mlp.0
t_embedder.mlp.2
```

All other source parameters and buffers retain BF16. Each component therefore contains 1,060 tensors with this dtype count:

```text
F32:  210
BF16: 430
I8:   210
U8:   210
```

The final file orders tensors into four homogeneous contiguous runs per component: F32, BF16, I8, then U8. This lets the native loader use four sequential reads per component instead of hundreds of small mmap faults.

## Build

```bash
python scripts/build-checkpoint.py \
  --output /path/to/trellis_2_int8_convrot.safetensors \
  --int8-backend /path/to/ComfyUI-INT8-Fast-ROCM \
  --device cuda
```

No weights are bundled with this repository. The builder downloads the pinned sources from Hugging Face or accepts explicit `--structure`, `--shape`, and `--texture` files. It verifies source SHA256 values and the backend Git revision, then embeds the exact v1 provenance metadata in the safetensors header. `--skip-source-hash` is test-only: it marks the output unverified, so the publication validator rejects it.

## Strict validation

```bash
python scripts/validate-checkpoint.py /path/to/trellis_2_int8_convrot.safetensors
```

The validator checks:

- exact v1 file-level provenance metadata;
- the exact 3,180-key manifest, including every tensor name, dtype, and shape;
- safetensors byte sizes, bounds, global contiguity, and exact file length;
- four homogeneous component runs in the expected order;
- all 210 weight/scale/quant-record triples per component;
- valid quant JSON, format, ConvRot flag, and group size;
- I8 weight rank, group-size divisibility, F32 row-scale shape, and optional BF16 bias shape.

Malformed JSON, renamed or unknown tensors, wrong shapes or provenance, missing scales, fake one-byte weights, mixed runs, gaps, overlaps, truncation, and trailing data are rejected.

## Published BitPoet four-component artifact

[`BitPoet/TRELLIS.2-int8-convrot`](https://huggingface.co/BitPoet/TRELLIS.2-int8-convrot) publishes the 5,253,048,192-byte checkpoint used for runtime validation, with SHA256 `66d269c1f874d38fe491a413e16944ff208a4ae348e01fc3e97b5531b52a7f3f`.

Its component routing is:

```text
model.structure_model.*  -> shared structure flow
model.img2shape_512.*    -> 512 image-to-shape
model.img2shape.*        -> 1024 image-to-shape
model.shape2txt.*        -> 1024 shape-to-texture
```

This supports 512 shape generation and complete 1024/1024-cascade flow routing. It does not contain a separate 512 texture component. `scripts/verify-bitpoet-checkpoint.py` verifies the exact community artifact; `validate-checkpoint.py` remains specific to the metadata-tagged three-component v1 rebuild.

## Distribution

Derived checkpoints remain subject to the source model license and hosting terms. Verify those terms before distributing a checkpoint; this repository distributes code and patches only.
