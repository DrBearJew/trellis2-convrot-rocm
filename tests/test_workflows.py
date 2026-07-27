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
        self.assertEqual(payload["2"]["class_type"], "LoadImage")
        self.assertTrue(all("GGUF" not in node["class_type"] for node in payload.values()))
        self.assertEqual(payload["4"]["inputs"]["pipeline_type"], "1024")
        self.assertFalse(payload["4"]["inputs"]["generate_texture_slat"])
        self.assertEqual(payload["6"]["inputs"]["file_format"], "glb")


if __name__ == "__main__":
    unittest.main()
