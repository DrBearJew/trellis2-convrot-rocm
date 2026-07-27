#!/usr/bin/env python3
"""Focused post-install checks for fallback and the 512-only invariant."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True, type=Path)
    parser.add_argument("--trellis-node", required=True, type=Path)
    parser.add_argument("--int8-backend", required=True, type=Path)
    args = parser.parse_args()

    sys.path[:0] = [os.fspath(args.trellis_node.resolve()), os.fspath(args.comfyui.resolve())]
    os.environ["TRELLIS2_INT8_FAST_PATH"] = os.fspath(args.int8_backend.resolve())

    from trellis2_gguf.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2_gguf.utils import convrot_utils

    layer = convrot_utils.ConvRotLinear(
        4, 3, bias=True, sparse=False, cache_context=False, group_size=4, device="cpu"
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor([[1, 2, 3, 4], [-2, 1, 0, 3], [4, -1, 2, 0]], dtype=torch.int8)
        )
        layer.weight_scale.fill_(0.125)
        layer.bias.copy_(torch.tensor([0.5, -0.25, 0.75], dtype=torch.bfloat16))
    original_loader = convrot_utils._load_triton_backend_modules
    convrot_utils._load_triton_backend_modules = lambda: (None, None)
    try:
        output = layer(torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.bfloat16))
    finally:
        convrot_utils._load_triton_backend_modules = original_loader
    assert output.shape == (1, 3) and torch.isfinite(output).all()

    pipeline = object.__new__(Trellis2ImageTo3DPipeline)
    pipeline.enable_convrot = True
    pipeline.default_pipeline_type = "512"
    unsupported = [
        ("load_shape_slat_flow_model_1024", (), {}),
        ("load_tex_slat_flow_model_1024", (), {}),
        ("run", (None,), {"pipeline_type": "1024"}),
        ("run_multiview", (None,), {"pipeline_type": "1024"}),
        ("run_cascade", (None,), {}),
        ("texture_mesh", (None, None), {}),
        ("texture_mesh_multiview", (None, None, None, None, None), {}),
        ("refine_mesh", (None, None), {}),
    ]
    for method, positional, keyword in unsupported:
        try:
            getattr(pipeline, method)(*positional, **keyword)
        except ValueError as exc:
            assert "gated to 512" in str(exc)
        else:
            raise AssertionError(f"{method} accepted unsupported ConvRot resolution")

    try:
        convrot_utils.checkpoint_prefix_for_path(
            "/models/slat_flow_img2shape_dit_1_3B_1024_bf16"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("1024 shape prefix was accepted")

    print("native contract: PASS")


if __name__ == "__main__":
    main()
