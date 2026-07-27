from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts/build-checkpoint.py"
VALIDATE = ROOT / "scripts/validate-checkpoint.py"
MANIFEST = ROOT / "manifests/trellis2-convrot-v1.json"
DTYPE_BYTES = {"F32": 4, "BF16": 2, "I8": 1, "U8": 1}
DTYPE_ORDER = ("F32", "BF16", "I8", "U8")
QUANT_RECORD = json.dumps(
    {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
).encode()


class CheckpointToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text())

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="checkpoint-tools-"))
        self.output = self.temp / "sparse.safetensors"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _write_sparse(
        self,
        mutate: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        manifest = self.manifest
        header: dict[str, object] = {"__metadata__": deepcopy(manifest["metadata"])}
        cursor = 0
        for dtype in DTYPE_ORDER:
            for prefix in manifest["required_prefixes"]:
                for key in sorted(
                    key
                    for key, schema in manifest["tensors"].items()
                    if key.startswith(prefix) and schema["dtype"] == dtype
                ):
                    schema = manifest["tensors"][key]
                    length = math.prod(schema["shape"]) * DTYPE_BYTES[dtype]
                    header[key] = {
                        "dtype": dtype,
                        "shape": list(schema["shape"]),
                        "data_offsets": [cursor, cursor + length],
                    }
                    cursor += length
        if mutate:
            mutate(header)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        with self.output.open("wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            data_start = 8 + len(encoded)
            for key, record in header.items():
                if key != "__metadata__" and key.endswith(".comfy_quant"):
                    start, end = record["data_offsets"]
                    self.assertEqual(end - start, len(QUANT_RECORD))
                    handle.seek(data_start + start)
                    handle.write(QUANT_RECORD)
            handle.truncate(data_start + cursor)

    def _validate(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VALIDATE), str(self.output)],
            text=True,
            capture_output=True,
            check=False,
        )

    def _assert_rejected(self, expected: str) -> None:
        result = self._validate()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stderr)

    def test_sparse_exact_manifest_passes(self) -> None:
        self._write_sparse()
        result = self._validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["contract"], "trellis2-convrot-v1")
        self.assertEqual(len(summary["components"]), 3)
        self.assertLess(self.output.stat().st_blocks * 512, self.output.stat().st_size)

    def test_unknown_tensor_rejected(self) -> None:
        def mutate(header: dict[str, object]) -> None:
            key = next(key for key in header if key != "__metadata__")
            header["model.unknown.weight"] = deepcopy(header[key])

        self._write_sparse(mutate)
        self._assert_rejected("tensor key set does not match manifest")

    def test_misnamed_tensor_rejected(self) -> None:
        def mutate(header: dict[str, object]) -> None:
            key = next(key for key in header if key != "__metadata__")
            header[key + ".typo"] = header.pop(key)

        self._write_sparse(mutate)
        self._assert_rejected("tensor key set does not match manifest")

    def test_wrong_shape_rejected(self) -> None:
        def mutate(header: dict[str, object]) -> None:
            key = next(key for key in header if key != "__metadata__")
            header[key]["shape"][0] += 1

        self._write_sparse(mutate)
        self._assert_rejected("!= manifest")

    def test_wrong_metadata_rejected(self) -> None:
        def mutate(header: dict[str, object]) -> None:
            header["__metadata__"]["backend_revision"] = "wrong"

        self._write_sparse(mutate)
        self._assert_rejected("checkpoint metadata does not match")

    def test_invalid_offsets_rejected(self) -> None:
        def mutate(header: dict[str, object]) -> None:
            key = next(key for key in header if key != "__metadata__")
            header[key]["data_offsets"][0] += 1

        self._write_sparse(mutate)
        self._assert_rejected("byte-size mismatch")

    def test_invalid_quant_json_rejected(self) -> None:
        self._write_sparse()
        with self.output.open("r+b") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            key = next(key for key in header if key.endswith(".comfy_quant"))
            handle.seek(8 + header_size + header[key]["data_offsets"][0])
            handle.write(b"x")
        self._assert_rejected("invalid comfy_quant JSON")


class BackendProvenanceTest(unittest.TestCase):
    def test_modified_convrot_is_rejected(self) -> None:
        backend = Path(os.environ.get("TRELLIS_TEST_INT8_BACKEND", ""))
        convrot = backend / "convrot.py"
        if not convrot.is_file():
            self.skipTest("set TRELLIS_TEST_INT8_BACKEND to the pinned backend checkout")
        spec = importlib.util.spec_from_file_location("checkpoint_builder_test", BUILD)
        assert spec and spec.loader
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        original = convrot.read_bytes()
        try:
            builder._verify_backend_revision(backend)
            convrot.write_bytes(original + b"\n# provenance mutation\n")
            with self.assertRaisesRegex(SystemExit, "differs from the pinned revision"):
                builder._verify_backend_revision(backend)
        finally:
            convrot.write_bytes(original)
        builder._verify_backend_revision(backend)


if __name__ == "__main__":
    unittest.main()
