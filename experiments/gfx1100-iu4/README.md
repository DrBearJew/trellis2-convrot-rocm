# gfx1100 native IU4 GEMM experiment

This is an isolated candidate, not a promoted TRELLIS runtime path.

It compares three exact signed-integer routes on an RX 7900 XTX:

- packed W4A4 through `v_wmma_i32_16x16x16_iu4`;
- matched W8A8 through `v_wmma_i32_16x16x16_iu8`;
- packed I8 DOT4 through `v_dot4_i32_i8` as a small/tail fallback control.

Both input matrices use row-major `[rows, K]` logical storage. INT4 stores two signed two's-complement nibbles per byte. The weight matrix is stored as `[N, K]`; the kernel computes `C[M,N] = A[M,K] * B[N,K]^T`. INT32 accumulation is exact.

The WMMA fragment contract reuses the locally proven Packed16 mapping:

- `lane & 15` selects the A row and B/output column;
- each lane contributes all 16 K values;
- accumulator register `i` maps to output row `2*i + lane_hi`.

## Build, assembly gate, and correctness

```bash
./build-and-verify.sh
```

The script fails unless the generated gfx1100 code object contains all three required instructions and all exact CPU-reference tests pass, including M/N/K tails.

## Benchmark

One shape:

```bash
./build/iu4_gemm_probe --shape 4096 1536 1536 --iterations 10 --repeats 5
```

Known TRELLIS projection shapes:

```bash
./build/iu4_gemm_probe --trellis-shapes --iterations 10 --repeats 5
```

Reported TOPS are effective GEMM operations (`2*M*N*K/time`). The candidate performs direct global-memory fragment loads and is a correctness/roofline probe, not yet an LDS-staged production GEMM. Promotion requires beating the existing tuned W8A8 implementation after activation rotation, quantization, packing, scaling, and output conversion are included.

## Intended dispatcher

- full aligned dominant tiles: IU4 WMMA;
- small/tail or protected W8 work: DOT4 I8 or IU8 WMMA, selected by measured shape;
- unsupported hardware or failed contract: existing runtime fallback.

Quality tuning remains numerically separable: the existing decode-to-IU8 W4A4 route can serve as an arithmetic oracle while native IU4 must match its packed integer result before integration.
