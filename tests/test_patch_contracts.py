from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRELLIS_PATCH = ROOT / "patches" / "trellis2-gguf-mit.patch"


class TrellisPatchContractTest(unittest.TestCase):
    def test_rocm_cumesh_preflight_preserves_safe_gpu_first_path(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn(
            'vertices = vertices.to(device="cuda", dtype=torch.float32).contiguous()',
            added,
        )
        self.assertIn(
            'faces = faces.to(device="cuda", dtype=torch.int32).contiguous()',
            added,
        )
        self.assertIn("rocm_vertex_row_limit = 1 << 20", added)
        self.assertIn(
            "if torch.version.hip is not None and int(vertices.shape[0]) >= rocm_vertex_row_limit:",
            added,
        )
        self.assertIn(
            "preinit_face_limit = max(int(target_face_num), 1_000_000)", added
        )
        self.assertIn("MeshUtils.simplify_with_meshlib(", added)
        self.assertIn("torch.cuda.synchronize()", added)
        self.assertNotIn("except RuntimeError as direct_init_error:", added)

    def test_hole_filling_prefers_cumesh_and_retains_cpu_fallback(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn(
            'cumesh.fill_holes(max_hole_perimeter=float("inf"))', added
        )
        self.assertIn("except RuntimeError as gpu_fill_error:", added)
        self.assertIn("falling back to Meshlib on CPU", added)


if __name__ == "__main__":
    unittest.main()
