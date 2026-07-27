#!/usr/bin/env python3
"""Strictly validate a TRELLIS.2 native ConvRot safetensors checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import struct
from typing import Any

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests/trellis2-convrot-v1.json"
DTYPE_BYTES = {"F32": 4, "BF16": 2, "I8": 1, "U8": 1}
EXPECTED_DTYPE_COUNTS = {"F32": 210, "BF16": 430, "I8": 210, "U8": 210}
EXPECTED_RUN_DTYPES = ("F32", "BF16", "I8", "U8")
EXPECTED_GROUP_SIZE = 256


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot load contract manifest {path}: {exc}")
    if not isinstance(manifest, dict):
        _fail(f"contract manifest {path} must be an object")
    return manifest


class ContractError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ContractError(message)


def _shape(record: dict[str, Any], key: str) -> tuple[int, ...]:
    shape = record.get("shape")
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        _fail(f"{key}: invalid shape {shape!r}")
    return tuple(shape)


def _range(record: dict[str, Any], key: str, data_size: int) -> tuple[int, int]:
    offsets = record.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int for value in offsets)
    ):
        _fail(f"{key}: invalid data_offsets {offsets!r}")
    start, end = offsets
    if start < 0 or end < start or end > data_size:
        _fail(f"{key}: out-of-bounds range {start}:{end}")
    return start, end


def _validate_record(record: Any, key: str, data_size: int) -> tuple[int, int]:
    if not isinstance(record, dict):
        _fail(f"{key}: tensor record must be an object")
    dtype = record.get("dtype")
    if dtype not in DTYPE_BYTES:
        _fail(f"{key}: unsupported dtype {dtype!r}")
    shape = _shape(record, key)
    start, end = _range(record, key, data_size)
    expected = math.prod(shape) * DTYPE_BYTES[dtype]
    if end - start != expected:
        _fail(f"{key}: byte-size mismatch, range={end - start}, shape/dtype={expected}")
    return start, end


def _read_quant_record(handle, data_start: int, key: str, record: dict[str, Any]) -> dict[str, Any]:
    if record.get("dtype") != "U8" or len(_shape(record, key)) != 1:
        _fail(f"{key}: comfy_quant must be a one-dimensional U8 tensor")
    start, end = record["data_offsets"]
    handle.seek(data_start + start)
    raw = handle.read(end - start)
    if len(raw) != end - start:
        _fail(f"{key}: short read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        _fail(f"{key}: invalid comfy_quant JSON: {exc}")
    if not isinstance(value, dict):
        _fail(f"{key}: comfy_quant JSON must be an object")
    if value.get("format") != "int8_tensorwise" or value.get("convrot") is not True:
        _fail(f"{key}: unsupported quantization record {value!r}")
    if value.get("convrot_groupsize") != EXPECTED_GROUP_SIZE:
        _fail(f"{key}: expected convrot_groupsize={EXPECTED_GROUP_SIZE}, found {value.get('convrot_groupsize')!r}")
    return value


def _validate_component(handle, data_start: int, records: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    items = sorted(
        (record["data_offsets"][0], record["data_offsets"][1], key, record)
        for key, record in records.items()
        if key.startswith(prefix)
    )
    if not items:
        _fail(f"missing component {prefix}")

    dtype_counts = Counter(item[3]["dtype"] for item in items)
    if dict(dtype_counts) != EXPECTED_DTYPE_COUNTS:
        _fail(f"{prefix}: dtype counts {dict(dtype_counts)} != {EXPECTED_DTYPE_COUNTS}")

    runs: list[list[tuple[int, int, str, dict[str, Any]]]] = []
    current = [items[0]]
    for item in items[1:]:
        if item[0] == current[-1][1]:
            current.append(item)
        else:
            runs.append(current)
            current = [item]
    runs.append(current)
    run_dtypes = tuple(run[0][3]["dtype"] for run in runs)
    if len(runs) != 4 or run_dtypes != EXPECTED_RUN_DTYPES:
        _fail(f"{prefix}: expected contiguous runs {EXPECTED_RUN_DTYPES}, found {run_dtypes}")
    for run in runs:
        if len({item[3]["dtype"] for item in run}) != 1:
            _fail(f"{prefix}: a contiguous run mixes dtypes")

    quant_keys = [key for _, _, key, _ in items if key.endswith(".comfy_quant")]
    if len(quant_keys) != 210:
        _fail(f"{prefix}: expected 210 comfy_quant records, found {len(quant_keys)}")

    for quant_key in quant_keys:
        module = quant_key[: -len(".comfy_quant")]
        weight_key = module + ".weight"
        scale_key = module + ".weight_scale"
        try:
            quant = records[quant_key]
            weight = records[weight_key]
            scale = records[scale_key]
        except KeyError as exc:
            _fail(f"{quant_key}: missing paired tensor {exc.args[0]}")
        _read_quant_record(handle, data_start, quant_key, quant)
        weight_shape = _shape(weight, weight_key)
        scale_shape = _shape(scale, scale_key)
        if weight.get("dtype") != "I8" or len(weight_shape) != 2:
            _fail(f"{weight_key}: expected two-dimensional I8 weight")
        if weight_shape[1] % EXPECTED_GROUP_SIZE:
            _fail(f"{weight_key}: in_features {weight_shape[1]} is not divisible by {EXPECTED_GROUP_SIZE}")
        if scale.get("dtype") != "F32" or scale_shape != (weight_shape[0], 1):
            _fail(f"{scale_key}: expected F32 shape {(weight_shape[0], 1)}, found {scale.get('dtype')} {scale_shape}")
        bias_key = module + ".bias"
        if bias_key in records:
            bias = records[bias_key]
            if bias.get("dtype") != "BF16" or _shape(bias, bias_key) != (weight_shape[0],):
                _fail(f"{bias_key}: expected BF16 shape {(weight_shape[0],)}")

    return {
        "tensors": len(items),
        "convrot_layers": len(quant_keys),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "payload_bytes": sum(end - start for start, end, _, _ in items),
        "run_dtypes": list(run_dtypes),
    }


def validate(path: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    required_prefixes = tuple(manifest.get("required_prefixes", ()))
    expected_tensors = manifest.get("tensors")
    expected_metadata = manifest.get("metadata")
    if not required_prefixes or not isinstance(expected_tensors, dict):
        _fail(f"contract manifest {manifest_path} is missing required prefixes or tensors")
    if not isinstance(expected_metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in expected_metadata.items()
    ):
        _fail(f"contract manifest {manifest_path} has invalid metadata")
    size = path.stat().st_size
    with path.open("rb", buffering=0) as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            _fail("invalid safetensors header")
        header_size = struct.unpack("<Q", raw)[0]
        if not 0 < header_size < size - 8:
            _fail(f"invalid header size: {header_size}")
        try:
            header = json.loads(handle.read(header_size))
        except Exception as exc:
            _fail(f"invalid safetensors JSON header: {exc}")
        if not isinstance(header, dict):
            _fail("safetensors header must be an object")
        metadata = header.get("__metadata__")
        if metadata != expected_metadata:
            _fail(f"checkpoint metadata does not match {manifest.get('contract')}: {metadata!r}")
        records = {key: value for key, value in header.items() if key != "__metadata__"}
        actual_keys = set(records)
        expected_keys = set(expected_tensors)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)[:5]
            extra = sorted(actual_keys - expected_keys)[:5]
            _fail(f"tensor key set does not match manifest: missing={missing}, extra={extra}")
        for key, expected in expected_tensors.items():
            record = records[key]
            if not isinstance(record, dict):
                _fail(f"{key}: tensor record must be an object")
            actual_schema = {"dtype": record.get("dtype"), "shape": record.get("shape")}
            if actual_schema != expected:
                _fail(f"{key}: schema {actual_schema!r} != manifest {expected!r}")
        data_size = size - 8 - header_size
        ranges = []
        for key, record in records.items():
            start, end = _validate_record(record, key, data_size)
            ranges.append((start, end, key))
        ranges.sort()
        cursor = 0
        for start, end, key in ranges:
            if start != cursor:
                _fail(f"{key}: non-contiguous or overlapping global range at {start}, expected {cursor}")
            cursor = end
        if cursor != data_size:
            _fail(f"checkpoint has trailing data: tensors end at {cursor}, payload is {data_size}")

        summary = {
            prefix: _validate_component(handle, 8 + header_size, records, prefix)
            for prefix in required_prefixes
        }

    return {
        "path": os.fspath(path),
        "bytes": size,
        "contract": manifest["contract"],
        "manifest": os.fspath(manifest_path),
        "required_components": list(required_prefixes),
        "components": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        result = validate(args.checkpoint, args.manifest)
    except (OSError, ContractError) as exc:
        raise SystemExit(f"checkpoint contract failed: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
