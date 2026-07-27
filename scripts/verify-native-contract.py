#!/usr/bin/env python3
"""Focused post-install checks for fallback and checkpoint-aware routing."""

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

    from trellis2_gguf.utils import convrot_utils, uv_rasterizer

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

    vertices = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    mask, positions = uv_rasterizer.rasterize_uv_positions(
        vertices, faces, vertices[:, :2], 8, face_chunk_size=1
    )
    assert mask.all() and positions.shape == (64, 3)

    checkpoint = convrot_utils.get_convrot_checkpoint()
    components, contract = convrot_utils._checkpoint_layout(os.path.realpath(checkpoint))
    assert convrot_utils.checkpoint_prefix_for_path(
        "/models/ss_flow_img_dit_1_3B_64_bf16", checkpoint
    ) == "structure_model"
    assert convrot_utils.checkpoint_prefix_for_path(
        "/models/slat_flow_img2shape_dit_1_3B_512_bf16", checkpoint
    ) == "img2shape_512"
    if contract == "trellis2-convrot-v1":
        assert convrot_utils.checkpoint_prefix_for_path(
            "/models/slat_flow_imgshape2tex_dit_1_3B_512_bf16", checkpoint
        ) == "shape2txt"
    else:
        assert {"img2shape", "shape2txt"}.issubset(components)
        assert convrot_utils.checkpoint_prefix_for_path(
            "/models/slat_flow_img2shape_dit_1_3B_1024_bf16", checkpoint
        ) == "img2shape"
        assert convrot_utils.checkpoint_prefix_for_path(
            "/models/slat_flow_imgshape2tex_dit_1_3B_1024_bf16", checkpoint
        ) == "shape2txt"

    print("native contract: PASS")


if __name__ == "__main__":
    main()
