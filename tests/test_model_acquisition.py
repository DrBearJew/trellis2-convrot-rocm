from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest


PIPELINE_MODELS = {
    "sparse_structure_decoder": "ckpts/ss_dec_conv3d_16l8_fp16",
    "sparse_structure_flow_model": "ckpts/ss_flow_img_dit_1_3B_64_bf16",
    "shape_slat_decoder": "ckpts/shape_dec_next_dc_f16c32_fp16",
    "shape_slat_flow_model_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
    "shape_slat_flow_model_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
    "tex_slat_decoder": "ckpts/tex_dec_next_dc_f16c32_fp16",
    "tex_slat_flow_model_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "tex_slat_flow_model_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
}


class ConvRotAcquisitionTest(unittest.TestCase):
    def setUp(self) -> None:
        node_root = os.environ.get("TRELLIS_TEST_NODE", "")
        manager_path = Path(node_root) / "model_manager.py"
        if not manager_path.is_file():
            self.skipTest("set TRELLIS_TEST_NODE to the patched node checkout")

        self.temp = tempfile.TemporaryDirectory(prefix="convrot-acquisition-")
        self.models_dir = Path(self.temp.name) / "models"
        self.downloads: list[tuple[str, str]] = []

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(self.models_dir)
        sys.modules["folder_paths"] = folder_paths

        hub = types.ModuleType("huggingface_hub")
        hub.__path__ = []
        utils = types.ModuleType("huggingface_hub.utils")
        file_download = types.ModuleType("huggingface_hub.file_download")
        constants = types.ModuleType("huggingface_hub.constants")

        def fake_download(*, repo_id: str, filename: str, local_dir: str, **_kwargs):
            self.downloads.append((repo_id, filename))
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}" if target.suffix == ".json" else "fixture")
            return str(target)

        hub.hf_hub_download = fake_download
        hub.utils = utils
        hub.file_download = file_download
        hub.constants = constants
        sys.modules["huggingface_hub"] = hub
        sys.modules["huggingface_hub.utils"] = utils
        sys.modules["huggingface_hub.file_download"] = file_download
        sys.modules["huggingface_hub.constants"] = constants

        spec = importlib.util.spec_from_file_location("convrot_test_model_manager", manager_path)
        assert spec and spec.loader
        self.manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.manager)

    def tearDown(self) -> None:
        for name in (
            "folder_paths",
            "huggingface_hub",
            "huggingface_hub.utils",
            "huggingface_hub.file_download",
            "huggingface_hub.constants",
        ):
            sys.modules.pop(name, None)
        self.temp.cleanup()

    def test_clean_root_acquires_ancillary_assets_without_flow_weights(self) -> None:
        old = os.environ.get("TRELLIS2_MODEL_RESOLUTION")
        os.environ["TRELLIS2_MODEL_RESOLUTION"] = "512"
        try:
            resolved = self.manager.ensure_model_files(
                "INT8 ConvRot",
                {"args": {"models": PIPELINE_MODELS}},
                gguf_repo="Aero-Ex/Trellis2-GGUF",
            )
        finally:
            if old is None:
                os.environ.pop("TRELLIS2_MODEL_RESOLUTION", None)
            else:
                os.environ["TRELLIS2_MODEL_RESOLUTION"] = old

        filenames = {filename for _repo, filename in self.downloads}
        flow_bases = ("ss_flow_", "slat_flow_img2shape_", "slat_flow_imgshape2tex_")
        flow_payloads = [
            name for name in filenames
            if name.endswith((".safetensors", ".gguf"))
            and Path(name).name.startswith(flow_bases)
        ]
        self.assertEqual(flow_payloads, [])
        self.assertIn("refiner/ss_flow_img_dit_1_3B_64_bf16.json", filenames)
        self.assertIn("shape/slat_flow_img2shape_dit_1_3B_512_bf16.json", filenames)
        self.assertIn("texture/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", filenames)
        self.assertFalse(any("_1024_" in name for name in filenames))
        self.assertIn("decoders/Stage1/ss_dec_conv3d_16l8_fp16.safetensors", filenames)
        self.assertIn("decoders/Stage2/shape_dec_next_dc_f16c32_fp16.safetensors", filenames)
        self.assertIn("decoders/Stage2/tex_dec_next_dc_f16c32_fp16.safetensors", filenames)
        self.assertEqual(sum(repo == self.manager.DINOV3_REPO for repo, _ in self.downloads), 3)
        self.assertIsNone(resolved["sparse_structure_flow_model"][1])
        self.assertIsNone(resolved["shape_slat_flow_model_512"][1])
        self.assertIsNone(resolved["tex_slat_flow_model_512"][1])

    def test_convrot_rejects_non_512_resolution(self) -> None:
        old = os.environ.get("TRELLIS2_MODEL_RESOLUTION")
        os.environ["TRELLIS2_MODEL_RESOLUTION"] = "1024"
        try:
            with self.assertRaisesRegex(ValueError, "512 only"):
                self.manager.ensure_model_files(
                    "INT8 ConvRot",
                    {"args": {"models": PIPELINE_MODELS}},
                )
        finally:
            if old is None:
                os.environ.pop("TRELLIS2_MODEL_RESOLUTION", None)
            else:
                os.environ["TRELLIS2_MODEL_RESOLUTION"] = old


if __name__ == "__main__":
    unittest.main()
