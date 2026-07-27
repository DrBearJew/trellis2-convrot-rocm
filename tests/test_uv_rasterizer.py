import importlib.util
import os
import unittest
from pathlib import Path

import torch


NODE_ROOT = os.environ.get("TRELLIS_TEST_NODE")


@unittest.skipUnless(NODE_ROOT, "set TRELLIS_TEST_NODE to the patched node checkout")
class UVRasterizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(NODE_ROOT) / "trellis2_gguf" / "utils" / "uv_rasterizer.py"
        spec = importlib.util.spec_from_file_location("trellis2_uv_rasterizer", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_square_covers_texture_and_interpolates_pixel_centers(self):
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        uvs = vertices[:, :2].clone()
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)

        mask, positions = self.module.rasterize_uv_positions(
            vertices, faces, uvs, 8, face_chunk_size=1
        )

        self.assertTrue(mask.all())
        expected = torch.stack(torch.meshgrid(
            (torch.arange(8) + 0.5) / 8,
            (torch.arange(8) + 0.5) / 8,
            indexing="xy",
        ), dim=-1).reshape(-1, 2)
        torch.testing.assert_close(positions[:, :2], expected)
        torch.testing.assert_close(positions[:, 2], torch.zeros(64))

    def test_degenerate_triangle_has_no_coverage(self):
        vertices = torch.zeros((3, 3))
        uvs = torch.zeros((3, 2))
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        mask, positions = self.module.rasterize_uv_positions(vertices, faces, uvs, 8)

        self.assertFalse(mask.any())
        self.assertEqual(tuple(positions.shape), (0, 3))


if __name__ == "__main__":
    unittest.main()
