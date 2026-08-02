from __future__ import annotations

import ast
import os
import types
import unittest
from pathlib import Path

from PIL import Image

NODE_ROOT_ENV = os.environ.get("TRELLIS_TEST_NODE")
NODE_ROOT = Path(NODE_ROOT_ENV or ".")
PIPELINE = NODE_ROOT / "trellis2_gguf/pipelines/trellis2_image_to_3d.py"


class _FakeTensor:
    def __init__(self, views: int):
        self.shape = (views, 3, 8, 8)
        self.ndim = 4


def _load_function(name: str):
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Trellis2ImageTo3DPipeline"
    )
    function = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    function.returns = None
    for argument in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        argument.annotation = None
    namespace = {
        "_require_single_projected_view": _load_guard(),
        "torch": types.SimpleNamespace(Tensor=_FakeTensor),
        "Image": Image,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE), "exec"), namespace)
    return namespace[name]


def _load_guard():
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_require_single_projected_view"
    )
    namespace = {"torch": types.SimpleNamespace(Tensor=_FakeTensor)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE), "exec"), namespace)
    return namespace["_require_single_projected_view"]


@unittest.skipUnless(NODE_ROOT_ENV, "set TRELLIS_TEST_NODE to an applied node checkout")
class ProjectedViewContractTest(unittest.TestCase):
    def test_projected_conditioning_rejects_unsupported_view_counts(self):
        guard = _load_guard()
        guard([object()])
        guard(_FakeTensor(1))
        for value, count in (([], 0), ([object(), object()], 2), (_FakeTensor(3), 3)):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, rf"exactly one view; received {count}"):
                    guard(value)

    def test_public_conditioning_and_entry_points_fail_before_model_access(self):
        get_cond = _load_function("get_cond")
        fake_pipeline = types.SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "received 2 views with max_views=1"):
            get_cond(
                fake_pipeline,
                [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
                512,
                max_views=1,
            )
        pixel3d = types.SimpleNamespace(isPixal3D=True, default_pipeline_type="1024")
        with self.assertRaisesRegex(ValueError, "exactly one view"):
            _load_function("run")(pixel3d, [object(), object()])
        with self.assertRaisesRegex(ValueError, "exactly one view"):
            _load_function("run_cascade")(pixel3d, [object(), object()])
        with self.assertRaisesRegex(ValueError, "Named multi-view generation"):
            _load_function("run_multiview")(pixel3d, object())

    def test_all_projected_conditioning_surfaces_are_guarded(self):
        source = PIPELINE.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("_require_single_projected_view("), 5)
        self.assertNotIn("range(min(int(image.shape[0]), max_views))", source)
        self.assertNotIn("list(image)[:max_views]", source)
        self.assertIn(
            "Named multi-view generation does not support projected Pixel3D conditioning",
            source,
        )
        init_source = (NODE_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("Projected Pixel3D conditioning currently supports exactly one view", init_source)


if __name__ == "__main__":
    unittest.main()
