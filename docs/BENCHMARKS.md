# Performance observations

These numbers are retained as transparent engineering observations. They are single runs, not a statistically controlled benchmark, and no confidence interval is available.

Machine-readable values are in [`benchmarks/observations.json`](../benchmarks/observations.json).

## Test system

- AMD Radeon RX 7900 XTX (`gfx1100`)
- Python 3.12.12
- PyTorch 2.14.0a0 ROCm 7.15 development build
- Triton 3.8.0
- 512 shape-only workflow
- 8 sparse-structure steps and 8 shape steps

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

## Why no end-to-end speedup is claimed

The recorded complete runs were:

| Format | Time | Vertices | Faces |
|---|---:|---:|---:|
| Q4_K_M | 131.67 s | 960,471 | 1,978,942 |
| INT8 ConvRot | 104.04 s | 811,704 | 1,710,874 |

The INT8 output had 15.5% fewer vertices and 13.5% fewer faces. Mesh decoding, extraction, simplification, and export dominate the full workflow, so dividing these elapsed times would confound model execution with output complexity. The earlier 1.27× label has therefore been withdrawn.

## Single-process acceptance evidence

A later one-process acceptance sequence ran Krea, native TRELLIS, Mesa `GLSLShader`, then Krea again. Its TRELLIS stage completed in 122.01 seconds and produced a 28,075,300-byte GLB with 766,481 vertices and 1,573,060 faces. This demonstrates integration, not comparative performance.

## Reproduction status

This repository publishes the checkpoint builder, exact source revision, runtime patches, traced shapes, environment, and raw observation record. It does not yet publish an automated multi-run benchmark harness. Treat all timing values as provisional observations.
