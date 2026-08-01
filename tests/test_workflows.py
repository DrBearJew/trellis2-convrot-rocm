import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"


class WorkflowTest(unittest.TestCase):
    def test_gui_workflow_is_connected_and_ready_for_1024(self):
        workflow = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024.workflow.json").read_text()
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}

        self.assertEqual(workflow["version"], 0.4)
        self.assertEqual(len(nodes), 6)
        self.assertEqual(len(workflow["links"]), 5)
        self.assertEqual(nodes[1]["type"], "Trellis2LoadModel_ConvRot")
        self.assertEqual(nodes[1]["widgets_values"][1], "INT8 ConvRot")
        self.assertFalse(nodes[1]["widgets_values"][4])
        self.assertFalse(nodes[1]["widgets_values"][5])
        self.assertEqual(nodes[2]["type"], "LoadImage")
        self.assertTrue(all("GGUF" not in node["type"] for node in nodes.values()))
        self.assertTrue(all("GGUF" not in node.get("title", "") for node in nodes.values()))
        self.assertEqual(nodes[4]["widgets_values"], [
            271828, "fixed", "1024", 8, 8, 8, 49152, 32, 1,
            False, False, "euler",
        ])

        for link_id, source, source_slot, target, target_slot, link_type in workflow["links"]:
            self.assertIn(link_id, nodes[source]["outputs"][source_slot]["links"])
            self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)
            self.assertEqual(nodes[source]["outputs"][source_slot]["type"], link_type)
            self.assertEqual(nodes[target]["inputs"][target_slot]["type"], link_type)

    def test_api_payload_matches_gui_route(self):
        payload = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024.api.json").read_text()
        )
        self.assertEqual(payload["1"]["class_type"], "Trellis2LoadModel_ConvRot")
        self.assertEqual(payload["1"]["inputs"]["model_format"], "INT8 ConvRot")
        self.assertFalse(payload["1"]["inputs"]["low_vram"])
        self.assertFalse(payload["1"]["inputs"]["keep_models_loaded"])
        self.assertEqual(payload["2"]["class_type"], "LoadImage")
        self.assertTrue(all("GGUF" not in node["class_type"] for node in payload.values()))
        self.assertEqual(payload["4"]["inputs"]["pipeline_type"], "1024")
        self.assertFalse(payload["4"]["inputs"]["generate_texture_slat"])
        self.assertEqual(payload["6"]["inputs"]["file_format"], "glb")

    def test_textured_gui_workflow_has_complete_bake_chain(self):
        workflow = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured.workflow.json").read_text()
        )
        nodes = {node["id"]: node for node in workflow["nodes"]}

        self.assertEqual(len(nodes), 8)
        self.assertEqual(len(workflow["links"]), 8)
        self.assertTrue(all("GGUF" not in node["type"] for node in nodes.values()))
        self.assertEqual(nodes[2]["type"], "LoadImage")
        self.assertFalse(nodes[1]["widgets_values"][4])
        self.assertFalse(nodes[1]["widgets_values"][5])
        self.assertTrue(nodes[4]["widgets_values"][9])
        self.assertEqual(nodes[5]["type"], "Trellis2TextureBake")
        self.assertEqual(nodes[5]["widgets_values"], [
            60.0, 0, 1, 1, 2048, True, 1.0, 0.0, 1000000,
            "Cumesh", True, "OPAQUE", "512", True, False, False,
            False, "Xatlas", False,
        ])
        self.assertEqual(nodes[5]["inputs"][0]["type"], "MESHWITHVOXEL")
        self.assertEqual(nodes[5]["inputs"][1]["type"], "BVH")
        self.assertEqual(nodes[6]["inputs"][0]["type"], "TRIMESH")

    def test_textured_api_payload_preserves_voxel_attributes_until_bake(self):
        payload = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured.api.json").read_text()
        )

        self.assertTrue(all("GGUF" not in node["class_type"] for node in payload.values()))
        self.assertFalse(payload["1"]["inputs"]["low_vram"])
        self.assertFalse(payload["1"]["inputs"]["keep_models_loaded"])
        self.assertTrue(payload["4"]["inputs"]["generate_texture_slat"])
        self.assertEqual(payload["5"]["class_type"], "Trellis2TextureBake")
        self.assertEqual(
            payload["5"]["inputs"]["mesh_cluster_threshold_cone_half_angle_rad"],
            60.0,
        )
        self.assertEqual(payload["5"]["inputs"]["mesh"], ["4", 0])
        self.assertEqual(payload["5"]["inputs"]["bvh"], ["4", 1])
        self.assertFalse(payload["5"]["inputs"]["bake_on_vertices"])
        self.assertEqual(payload["6"]["inputs"]["trimesh"], ["5", 0])
        self.assertEqual(payload["7"]["inputs"]["images"], ["5", 1])
        self.assertEqual(payload["8"]["inputs"]["images"], ["5", 2])

    def test_textured_fast_profile_only_changes_uv_clustering(self):
        standard_api = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured.api.json").read_text()
        )
        fast_api = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured_fast.api.json").read_text()
        )
        angle_key = "mesh_cluster_threshold_cone_half_angle_rad"
        self.assertEqual(standard_api["5"]["inputs"].pop(angle_key), 60.0)
        self.assertEqual(fast_api["5"]["inputs"].pop(angle_key), 20.0)
        self.assertEqual(fast_api, standard_api)

        standard_gui = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured.workflow.json").read_text()
        )
        fast_gui = json.loads(
            (WORKFLOWS / "trellis2_convrot_bitpoet_1024_textured_fast.workflow.json").read_text()
        )
        standard_bake = next(node for node in standard_gui["nodes"] if node["id"] == 5)
        fast_bake = next(node for node in fast_gui["nodes"] if node["id"] == 5)
        self.assertEqual(standard_bake["widgets_values"][0], 60.0)
        self.assertEqual(fast_bake["widgets_values"][0], 20.0)
        self.assertEqual(fast_bake["widgets_values"][1:], standard_bake["widgets_values"][1:])
        self.assertIn("FAST UV (20°)", fast_bake["title"])


if __name__ == "__main__":
    unittest.main()
