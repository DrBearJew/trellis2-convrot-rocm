#!/usr/bin/env python3
"""Verify the pinned BitPoet four-component TRELLIS.2 ConvRot artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

REPO_ID = "BitPoet/TRELLIS.2-int8-convrot"
REVISION = "2f7cd18627fc89c9f238e63bdd0abb5b204d13c1"
FILENAME = "trellis_2_int8_convrot.safetensors"
EXPECTED_BYTES = 5_253_048_192
EXPECTED_SHA256 = "66d269c1f874d38fe491a413e16944ff208a4ae348e01fc3e97b5531b52a7f3f"
MANIFEST = Path(__file__).resolve().parents[1] / "manifests/trellis2-convrot-v1.json"
SHAPE_1024_PREFIX = "model.img2shape."
CHUNK = 16 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument(
        "--download-to",
        type=Path,
        help="download the pinned artifact into this directory before verification",
    )
    args = parser.parse_args()
    if args.download_to and args.checkpoint:
        parser.error("checkpoint and --download-to are mutually exclusive")
    if args.download_to:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            revision=REVISION,
            local_dir=args.download_to.resolve(),
        )).resolve()
    elif args.checkpoint:
        path = args.checkpoint.resolve()
    else:
        parser.error("provide a checkpoint or --download-to DIRECTORY")
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise SystemExit(f"BitPoet checkpoint size mismatch: expected {EXPECTED_BYTES}, found {size}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise SystemExit(
            f"BitPoet checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, found {actual_hash}"
        )

    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    records = {key: value for key, value in header.items() if key != "__metadata__"}
    expected = contract["tensors"]
    for key, schema in expected.items():
        record = records.get(key)
        actual = None if record is None else {
            "dtype": record.get("dtype"),
            "shape": record.get("shape"),
        }
        if actual != schema:
            raise SystemExit(f"BitPoet runtime tensor mismatch for {key}: {actual!r} != {schema!r}")
    allowed = tuple(contract["required_prefixes"]) + (SHAPE_1024_PREFIX,)
    unknown = [key for key in records if not key.startswith(allowed)]
    shape_1024_count = sum(key.startswith(SHAPE_1024_PREFIX) for key in records)
    if unknown or len(records) != 4_240 or shape_1024_count != 1_060:
        raise SystemExit(
            "BitPoet checkpoint structure mismatch: "
            f"tensors={len(records)}, shape_1024={shape_1024_count}, unknown={unknown[:5]}"
        )
    print(
        "BitPoet checkpoint: PASS "
        f"({size} bytes, SHA256 {actual_hash}, 4 routed ConvRot components)"
    )


if __name__ == "__main__":
    main()
