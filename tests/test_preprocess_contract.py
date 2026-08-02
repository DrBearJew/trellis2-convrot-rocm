import importlib.util
import os
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - minimal host without runtime dependencies
    np = None
    Image = None


NODE_ROOT = Path(os.environ.get("TRELLIS_TEST_NODE", ""))


def load_preprocessor():
    source_path = NODE_ROOT / "trellis2_gguf" / "utils" / "image_preprocess.py"
    spec = importlib.util.spec_from_file_location("trellis2_image_preprocess", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda image: module.crop_and_premultiply_rgba(image, maximum_size=2048)


@unittest.skipUnless(
    NODE_ROOT.is_dir() and np is not None and Image is not None,
    "set TRELLIS_TEST_NODE to the patched node checkout with NumPy and Pillow",
)
class PreprocessContractTest(unittest.TestCase):
    def setUp(self):
        self.preprocess = load_preprocessor()

    def test_rgb_input_is_supported(self):
        result = self.preprocess(Image.new("RGB", (4, 2), (120, 80, 40)))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, (4, 4))
        self.assertEqual(result.getpixel((2, 2)), (120, 80, 40))

    def test_semitransparent_foreground_is_retained(self):
        image = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
        image.putpixel((1, 1), (200, 100, 50, 64))
        result = self.preprocess(image)
        self.assertEqual(result.size, (1, 1))
        self.assertEqual(result.getpixel((0, 0)), (50, 25, 12))

    def test_fully_transparent_input_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "no visible pixels"):
            self.preprocess(Image.new("RGBA", (2, 2), (0, 0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
