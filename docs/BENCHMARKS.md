# Performance observations

These numbers are retained as transparent engineering observations. They are single runs, not a statistically controlled benchmark, and no confidence interval is available.

Machine-readable values are in [`benchmarks/observations.json`](../benchmarks/observations.json).

## Test system

- AMD Radeon RX 7900 XTX (`gfx1100`)
- Python 3.12.12
- PyTorch 2.14.0a0 ROCm 7.15 development build
- Triton 3.8.0
- 512 shape-only flow measurements
- 8 sparse-structure steps and 8 shape steps
- separate 1024 textured acceptance run with 8 texture steps and 2048² maps

## Instrumented flow execution

| Flow execution | Q4_K_M | INT8 ConvRot | Observed ratio |
|---|---:|---:|---:|
| Cold structure | 4.927 s | 1.600 s | 3.08× |
| Cold shape | 5.802 s | 2.146 s | 2.70× |
| Warm structure | 0.341 s | 0.243 s | 1.40× |
| Warm shape | 0.791 s | 0.679 s | 1.16× |

“Cold” means the first execution after kernel/cache initialization for the observed process; “warm” means a repeat in that process. The timings came from instrumented flow sections and synchronized execution during development. They should be independently reproduced before being used for purchasing or deployment decisions.

Traced dimensions included:

```text
structure: 4096 x 1536
shape:     8398 x 1536 and 8398 x 8192
texture:   2875 x 1536 and 2875 x 8192
cross conditioning: [1, 1029, 1024]
```

The static `gfx1100` Triton configurations cover known TRELLIS projection shapes and are opt-in from the native adapter. Unknown shapes and generic INT8 users retain autotuning.

## Storage and model-footprint tradeoff

INT8 ConvRot is a speed-oriented format, not a size reduction over Q4_K_M:

| 512 flow payloads | Q4_K_M | INT8 ConvRot | INT8/Q4 |
|---|---:|---:|---:|
| Structure + image-to-shape (shape-only) | 1,578,195,424 B | 2,626,141,808 B | 1.66× |
| Structure + image-to-shape + shape-to-texture | 2,367,637,248 B | 3,939,776,576 B file | 1.66× |

The ready-to-download BitPoet checkpoint is 5,253,048,192 bytes because it contains both 512 and 1024 image-to-shape components plus the 1024 texture flow. Runtime CPU/GPU peaks depend on component unloading and allocator behavior, but each INT8 component is also larger than its Q4_K_M counterpart. The speed observations come from fused W8A8 execution avoiding repeated Q4_K_M dequantization; they do not imply lower storage or memory use.

## End-to-end observation and geometry-parity caveat

The recorded complete runs were:

| Format | Time | Vertices | Faces |
|---|---:|---:|---:|
| Q4_K_M | 131.67 s | 960,471 | 1,978,942 |
| INT8 ConvRot | 104.04 s | 811,704 | 1,710,874 |

The observed elapsed-time ratio was `131.67 / 104.04 = 1.27×` in favor of INT8. Report that as an uncontrolled wall-clock observation, not an apples-to-apples benchmark.

More importantly, the INT8 output had 15.5% fewer vertices and 13.5% fewer faces. That proves the quantized flow produced materially different sparse occupancy/topology. Polygon count alone does not establish lower visual quality, but **geometry-quality parity has not been demonstrated**. A promotion-quality comparison still needs identical input/seed/settings plus rendered-view comparison and geometry metrics (for example Chamfer distance and F-score), ideally after normalizing extraction/simplification targets.

## 1024 route acceptance

With the BitPoet checkpoint, the native shape-only `pipeline_type=1024` route loaded `structure_model` and the legacy-named `img2shape` 1024 component, completed in 127.13 seconds, and exported a valid 30,466,568-byte GLB with 819,421 vertices and 1,719,392 faces.

The complete textured route additionally loaded `shape2txt`, decoded the PBR voxel attributes, remeshed to 485,188 faces, created a UV atlas, and baked 2048² base-color and metallic-roughness maps with the ROCm-safe PyTorch rasterizer. It completed in 213.95 seconds and exported a 25,616,976-byte GLB containing `TEXCOORD_0`, one material, two textures, and two embedded images. Blender 5.1 imported one mesh, one material, both 2048² images, and two populated image-texture nodes. This is a single acceptance run, not a comparative benchmark.

## Single-process acceptance evidence

A later one-process acceptance sequence ran Krea, native TRELLIS, Mesa `GLSLShader`, then Krea again. Its TRELLIS stage completed in 122.01 seconds and produced a 28,075,300-byte GLB with 766,481 vertices and 1,573,060 faces. This demonstrates integration, not comparative performance.

## RX 7900 XTX GPU-first 1024 observation

A same-input, same-seed (`1212101`), same-step warm comparison was run for the complete 1024 textured route. The baseline used `low_vram=true` and Meshlib CPU hole filling; the candidate used `low_vram=false`, kept `keep_models_loaded=false`, initialized CuMesh directly on the full contiguous GPU mesh, and used CuMesh GPU hole filling with CPU fallbacks retained.

| Route | Time | Generated vertices/faces before bake | Final GLB vertices/faces |
|---|---:|---:|---:|
| Conservative baseline | 250.61 s | 337,823 / 667,292 | 191,078 / 290,836 |
| GPU-first candidate | 184.14 s | 337,823 / 667,292 | 191,122 / 290,836 |

The observed reduction was 66.47 seconds (26.5%, or 1.36× baseline/candidate). Both GLBs imported as one finite, consistently wound mesh with no non-manifold edges and the same final face count. UV seam splitting and hole triangulation produced a 44-vertex difference, so bitwise topology parity is not claimed. A separate cold candidate run directly initialized and remeshed a real 866,682-vertex / 1,728,856-face TRELLIS mesh without invoking CPU pre-simplification. A later 50/50/30-step run produced 1,051,966 vertices / 2,095,944 faces and crossed the observed 2^20 vertex-row failure boundary; the revised preflight sends only that larger class through Meshlib before the first CuMesh call. These remain single-run observations, not a statistically controlled benchmark.

## Xatlas clustering observation

The explicit RX 7900 XTX **fast UV** textured profile requests a 20-degree CuMesh cone half-angle before Xatlas. The standard textured workflow and generic node default remain at the conservative 60 degrees. CuMesh creates more bounded GPU clusters, reducing Xatlas's superlinear CPU charting work.

| Seed / final mesh | 60° execution / Xatlas | 20° execution / Xatlas | Overall reduction |
|---|---:|---:|---:|
| `1212101`, 290,836 faces | 184.14 s / 15.45 s | 176.20 s / 2.54 s | 7.94 s (4.3%) |
| `1212102`, 715,240 faces | 307.86 s / 115 s | 256.32 s / 53 s | 51.54 s (16.7%) |

All faces were preserved. Both candidate GLBs were finite, consistently wound, and had no non-manifold edges. The 20-degree route increased UV-split vertices from 191,122 to 212,952 on the smaller mesh and from 481,911 to 490,292 on the dense mesh. On the dense same-seed pair, all 715,240 faces matched exactly in 3D; baked base-color centroid samples had RGB MAE 0.0079, p95 face MAE 0.0222, and 0.35% of faces above 0.10. The atlas layout is not bitwise-equivalent, and the smaller GLB grew by about 1.08 MB.

## Reproduction status

This repository publishes the checkpoint builder, exact source revision, runtime patches, traced shapes, environment, and raw observation record. It does not yet publish an automated multi-run benchmark harness. Treat all timing values as provisional observations.
