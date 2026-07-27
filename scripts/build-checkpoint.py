#!/usr/bin/env python3
"""Build the three-component TRELLIS.2 INT8 ConvRot checkpoint from BF16 sources."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
from typing import BinaryIO

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests/trellis2-convrot-v1.json"
with MANIFEST_PATH.open(encoding="utf-8") as _manifest_handle:
    CONTRACT = json.load(_manifest_handle)

SOURCE_REPO = CONTRACT["source"]["repo"]
SOURCE_REVISION = CONTRACT["source"]["revision"]
SOURCES = tuple(
    (item["component"], item["path"], item["bytes"], item["sha256"])
    for item in CONTRACT["source"]["files"]
)
CONTRACT_TENSORS = CONTRACT["tensors"]
CONTRACT_METADATA = CONTRACT["metadata"]
BACKEND_REVISION = CONTRACT["backend"]["revision"]
EXCLUDED_LINEAR_MODULES = {
    "adaLN_modulation.1",
    "input_layer",
    "out_layer",
    "t_embedder.mlp.0",
    "t_embedder.mlp.2",
}
GROUP_SIZE = CONTRACT["backend"]["group_size"]
QUANT_RECORD = json.dumps(
    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
).encode("utf-8")
DTYPE_ORDER = ("F32", "BF16", "I8", "U8")
COPY_CHUNK = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_backend_revision(backend: Path) -> None:
    result = subprocess.run(
        ["git", "-C", os.fspath(backend.resolve()), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = result.stdout.strip()
    if result.returncode or revision != BACKEND_REVISION:
        detail = result.stderr.strip() or revision or "not a Git checkout"
        raise SystemExit(
            f"INT8 backend must be revision {BACKEND_REVISION}; found {detail}"
        )


def _load_convrot(backend: Path):
    path = backend.resolve() / "convrot.py"
    if not path.is_file():
        raise FileNotFoundError(f"INT8 backend convrot.py not found: {path}")
    package_name = "_trellis_checkpoint_backend"
    package = types.ModuleType(package_name)
    package.__path__ = [os.fspath(path.parent)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.convrot", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _is_quantized_weight(key: str, tensor: torch.Tensor) -> bool:
    if not key.endswith(".weight") or tensor.ndim != 2:
        return False
    module = key[: -len(".weight")]
    return module not in EXCLUDED_LINEAR_MODULES


def _quantize_component(source: Path, destination: Path, device: torch.device, convrot) -> None:
    output: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != 640:
            raise ValueError(f"{source}: expected 640 source tensors, found {len(keys)}")
        quantized = 0
        hadamard = convrot.build_hadamard(GROUP_SIZE, device=device, dtype=torch.float32)
        for key in keys:
            tensor = handle.get_tensor(key)
            if not _is_quantized_weight(key, tensor):
                output[key] = tensor
                continue
            if tensor.shape[1] % GROUP_SIZE:
                raise ValueError(f"{key}: in_features {tensor.shape[1]} is not divisible by {GROUP_SIZE}")
            weight = tensor.to(device=device, dtype=torch.float32)
            rotated = convrot.rotate_weight(weight, hadamard, group_size=GROUP_SIZE)
            scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
            quant = rotated.mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
            output[key] = quant.cpu()
            output[key[: -len(".weight")] + ".weight_scale"] = scale.cpu()
            output[key[: -len(".weight")] + ".comfy_quant"] = torch.tensor(
                list(QUANT_RECORD), dtype=torch.uint8
            )
            quantized += 1
            del tensor, weight, rotated, scale, quant
        if quantized != 210:
            raise ValueError(f"{source}: expected 210 ConvRot weights, quantized {quantized}")
        save_file(
            output,
            destination,
            metadata={
                "format": "trellis2-convrot-w8a8",
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "convrot_groupsize": str(GROUP_SIZE),
            },
        )
    del output


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: invalid safetensors header")
        size = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(size))
    return 8 + size, header


def _copy_range(source: BinaryIO, destination: BinaryIO, start: int, length: int) -> None:
    source.seek(start)
    remaining = length
    while remaining:
        chunk = source.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise EOFError(f"short read with {remaining} bytes remaining")
        destination.write(chunk)
        remaining -= len(chunk)


def _combine(
    components: list[tuple[str, Path]], output: Path, metadata: dict[str, str]
) -> None:
    headers = {}
    data_starts = {}
    for name, path in components:
        data_starts[name], headers[name] = _read_header(path)

    copies = []
    output_header: dict[str, dict] = {"__metadata__": metadata}
    cursor = 0
    for dtype in DTYPE_ORDER:
        for name, path in components:
            records = headers[name]
            keys = sorted(
                key for key, record in records.items()
                if key != "__metadata__" and record["dtype"] == dtype
            )
            for key in keys:
                record = records[key]
                old_start, old_end = record["data_offsets"]
                length = old_end - old_start
                new_key = f"model.{name}.{key}"
                output_header[new_key] = {
                    "dtype": record["dtype"],
                    "shape": record["shape"],
                    "data_offsets": [cursor, cursor + length],
                }
                copies.append((path, data_starts[name] + old_start, length))
                cursor += length

    actual_contract = {
        key: {"dtype": record["dtype"], "shape": record["shape"]}
        for key, record in output_header.items()
        if key != "__metadata__"
    }
    if actual_contract != CONTRACT_TENSORS:
        actual_keys = set(actual_contract)
        expected_keys = set(CONTRACT_TENSORS)
        missing = sorted(expected_keys - actual_keys)[:5]
        extra = sorted(actual_keys - expected_keys)[:5]
        mismatched = sorted(
            key
            for key in actual_keys & expected_keys
            if actual_contract[key] != CONTRACT_TENSORS[key]
        )[:5]
        raise ValueError(
            "built tensors do not match trellis2-convrot-v1 manifest: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    header_bytes = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        with temporary.open("wb") as destination, ExitStack() as stack:
            destination.write(struct.pack("<Q", len(header_bytes)))
            destination.write(header_bytes)
            handles = {path: stack.enter_context(path.open("rb")) for _, path in components}
            for path, start, length in copies:
                _copy_range(handles[path], destination, start, length)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--int8-backend", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--shape", type=Path)
    parser.add_argument("--texture", type=Path)
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="test-only escape hatch for synthetic source fixtures",
    )
    args = parser.parse_args()
    overrides = (args.structure, args.shape, args.texture)
    if any(overrides) and not all(overrides):
        parser.error("--structure, --shape, and --texture must be supplied together")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if all(overrides):
        source_paths = [path.resolve() for path in overrides]
    else:
        from huggingface_hub import hf_hub_download
        source_paths = [
            Path(hf_hub_download(
                SOURCE_REPO,
                filename,
                revision=SOURCE_REVISION,
                cache_dir=os.fspath(args.cache_dir) if args.cache_dir else None,
            ))
            for _, filename, _, _ in SOURCES
        ]

    if not args.skip_source_hash:
        for source, (_, filename, expected_size, expected_hash) in zip(source_paths, SOURCES, strict=True):
            actual_size = source.stat().st_size
            if actual_size != expected_size:
                raise SystemExit(f"{filename}: expected {expected_size} bytes, found {actual_size}")
            actual_hash = _sha256(source)
            if actual_hash != expected_hash:
                raise SystemExit(f"{filename}: SHA256 mismatch: {actual_hash}")

    if not args.skip_source_hash:
        _verify_backend_revision(args.int8_backend)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA-style ROCm device requested but torch.cuda.is_available() is false")
    convrot = _load_convrot(args.int8_backend)
    with tempfile.TemporaryDirectory(prefix="trellis2-convrot-", dir=output.parent) as temp:
        temporary_root = Path(temp)
        components = []
        for (name, _, _, _), source in zip(SOURCES, source_paths, strict=True):
            component = temporary_root / f"{name}.safetensors"
            print(f"quantizing {name}: {source}", flush=True)
            _quantize_component(source, component, device, convrot)
            components.append((name, component))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"combining three components: {output}", flush=True)
        metadata = dict(CONTRACT_METADATA)
        if args.skip_source_hash:
            metadata.update(
                {
                    "source_repo": "unverified-test-fixture",
                    "source_revision": "unverified",
                    "source_sha256": json.dumps(
                        {
                            filename: _sha256(source)
                            for source, (_, filename, _, _) in zip(
                                source_paths, SOURCES, strict=True
                            )
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "backend_revision": "unverified",
                }
            )
        _combine(components, output, metadata)
    print(f"built {output} ({output.stat().st_size} bytes)")
    print(f"validate with: {Path(__file__).with_name('validate-checkpoint.py')} {output}")


if __name__ == "__main__":
    main()
