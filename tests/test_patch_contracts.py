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

    def test_glb_export_preserves_normals_for_blender(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn('"use_custom_normals":("BOOLEAN",{"default":True})', added)
        self.assertGreaterEqual(added.count("include_normals=True"), 2)
        self.assertGreaterEqual(added.count("unitize_normals=True"), 2)
        self.assertIn('name="TRELLIS_PBR_MetallicRoughness"', added)
        self.assertIn('("baseColorTexture", "TRELLIS_BaseColor")', added)
        self.assertIn(
            '("metallicRoughnessTexture", "TRELLIS_MetallicRoughness")', added
        )
        self.assertGreaterEqual(
            added.count("tree_postprocessor=_trellis_glb_tree_postprocessor"), 2
        )
        self.assertIn('"trellis_trimesh_version": str(Trimesh.__version__)', added)
        self.assertIn("trimesh==4.9.0", added)
        self.assertIn('"trellis_coordinate_space": "normalized"', added)
        self.assertIn('"trellis_physical_scale_authorized": False', added)
        self.assertIn('"trellis_normal_semantic": "NORMAL"', added)
        self.assertIn('"trellis_metallic_channel": "B"', added)
        self.assertIn('"trellis_roughness_channel": "G"', added)

    def test_direct_texturing_reuses_rocm_safe_uv_path(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertGreaterEqual(
            added.count("from ..utils.uv_rasterizer import rasterize_uv_positions"),
            2,
        )
        self.assertGreaterEqual(
            added.count("mask, valid_pos = rasterize_uv_positions("), 2
        )
        self.assertGreaterEqual(
            added.count("vertices_torch.shape[0] >= 1 << 20"), 2
        )
        self.assertIn("z_proj = torch.cat([z_proj_lr, z_proj_hr], dim=-1)", added)

    def test_rocm_uv_rasterizer_bounds_pixel_candidates(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn("candidate_chunk_size: int = 1 << 20", added)
        self.assertIn("torch.searchsorted", added)
        self.assertIn(
            "for candidate_start in range(0, candidate_total, candidate_chunk_size)",
            added,
        )

    def test_preprocessor_handles_rgb_and_semitransparent_inputs(self) -> None:
        patch = TRELLIS_PATCH.read_text()
        added = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )

        self.assertIn("def _preprocess_trellis_image", added)
        self.assertIn('input_image = input_image.convert("RGBA")', added)
        self.assertIn('alpha_bbox = input_image.getchannel("A").getbbox()', added)
        self.assertIn("TRELLIS preprocessing found no visible pixels", added)
        self.assertNotIn("alpha = output_np[:, :, 3]", added)
        self.assertNotIn("alpha > 0.8 * 255", added)

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
