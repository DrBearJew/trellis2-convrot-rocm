from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRELLIS_PATCH = ROOT / "patches" / "trellis2-gguf-mit.patch"


class TrellisPatchContractTest(unittest.TestCase):
    def test_rocm_cumesh_preflight_is_bounded_and_cuda_scoped(self) -> None:
        patch = TRELLIS_PATCH.read_text()

        self.assertIn(
            "if torch.version.hip is not None and int(faces.shape[0]) > 1_000_000:",
            patch,
        )
        self.assertIn("preinit_face_limit = 1_000_000", patch)
        self.assertIn("MeshUtils.simplify_with_meshlib(", patch)
        self.assertIn(
            'vertices = vertices.to(device="cuda", dtype=torch.float32).contiguous()',
            patch,
        )
        self.assertIn(
            'faces = faces.to(device="cuda", dtype=torch.int32).contiguous()',
            patch,
        )
        self.assertNotIn(
            "preinit_face_limit = max(int(target_face_num), 1_000_000)", patch
        )


if __name__ == "__main__":
    unittest.main()
