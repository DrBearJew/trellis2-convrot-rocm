from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest


MODEL_BASENAMES = {
    "structure": "ss_flow_img_dit_1_3B_64_bf16",
    "shape_512": "slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_1024": "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "texture_512": "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "texture_1024": "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}


class CheckpointRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        node_root = Path(os.environ.get("TRELLIS_TEST_NODE", ""))
        module_path = node_root / "trellis2_gguf/utils/convrot_utils.py"
        if not module_path.is_file():
            raise unittest.SkipTest("set TRELLIS_TEST_NODE to the patched node checkout")
        spec = importlib.util.spec_from_file_location("convrot_routing_test", module_path)
        assert spec and spec.loader
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="convrot-routing-")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _checkpoint(self, components: list[str], contract: str | None = None) -> Path:
        header: dict[str, object] = {}
        if contract:
            header["__metadata__"] = {"contract": contract}
        for component in components:
            header[f"model.{component}.sentinel"] = {
                "dtype": "U8",
                "shape": [0],
                "data_offsets": [0, 0],
            }
        encoded = json.dumps(header).encode()
        path = Path(self.temp.name) / "checkpoint.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
        return path

    def _route(self, kind: str, checkpoint: Path) -> str:
        return self.module.checkpoint_prefix_for_path(
            "/models/" + MODEL_BASENAMES[kind], str(checkpoint)
        )

    def test_bitpoet_routes_512_shape_and_full_1024(self) -> None:
        checkpoint = self._checkpoint(
            ["structure_model", "img2shape", "img2shape_512", "shape2txt"]
        )
        self.assertEqual(self._route("structure", checkpoint), "structure_model")
        self.assertEqual(self._route("shape_512", checkpoint), "img2shape_512")
        self.assertEqual(self._route("shape_1024", checkpoint), "img2shape")
        self.assertEqual(self._route("texture_1024", checkpoint), "shape2txt")
        with self.assertRaises(KeyError):
            self._route("texture_512", checkpoint)

    def test_rebuilt_v1_routes_full_512(self) -> None:
        checkpoint = self._checkpoint(
            ["structure_model", "img2shape_512", "shape2txt"],
            contract="trellis2-convrot-v1",
        )
        self.assertEqual(self._route("shape_512", checkpoint), "img2shape_512")
        self.assertEqual(self._route("texture_512", checkpoint), "shape2txt")
        with self.assertRaises(KeyError):
            self._route("shape_1024", checkpoint)
        with self.assertRaises(KeyError):
            self._route("texture_1024", checkpoint)


if __name__ == "__main__":
    unittest.main()
