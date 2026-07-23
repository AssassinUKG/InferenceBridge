import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


LAB_PATH = Path(__file__).resolve().parents[1] / "lab.py"
SPEC = importlib.util.spec_from_file_location("image_gen_lab", LAB_PATH)
assert SPEC and SPEC.loader
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = lab.load_json(lab.MANIFEST_PATH)

    def test_checked_in_manifest_is_valid(self):
        self.assertEqual([], lab.validate_manifest(self.manifest))

    def test_bundle_has_three_distinct_components(self):
        ids = lab.bundle_asset_ids(self.manifest, "qwen-2512-q6-quality")
        self.assertEqual(3, len(ids))
        self.assertEqual(3, len(set(ids)))

    def test_quality_profile_never_uses_q4(self):
        bundle = self.manifest["bundles"]["qwen-2512-q6-quality"]
        transformer = self.manifest["assets"][bundle["transformer"]]["filename"]
        self.assertIn("Q6_K", transformer)
        self.assertNotIn("Q4", transformer)

    def test_manifest_is_pinned_to_revisions(self):
        for asset in self.manifest["assets"].values():
            self.assertRegex(asset["revision"], r"^[0-9a-f]{40}$")
            self.assertIn(asset["revision"], asset["url"])

    def test_cuda_runtime_is_pinned(self):
        cuda_runtime = self.manifest["runtime"]["cuda_runtime"]
        self.assertRegex(cuda_runtime["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("cuda", cuda_runtime["asset_name"])


class SafetyTests(unittest.TestCase):
    def test_resolve_under_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            with self.assertRaises(lab.LabError):
                lab.resolve_under(root, "../escape.bin")

    def test_safe_zip_path_logic_rejects_external_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            safe = lab.resolve_under(root, "models/model.gguf")
            self.assertTrue(str(safe).startswith(str(root.resolve())))


class RunnerArgumentTests(unittest.TestCase):
    def setUp(self):
        self.manifest = lab.load_json(lab.MANIFEST_PATH)

    def test_quality_arguments_have_expected_safety_and_quality_flags(self):
        args = lab.build_sd_arguments(
            self.manifest,
            executable=Path("sd-cli.exe"),
            bundle_name="qwen-2512-q6-quality",
            profile_name="quality-1024",
            prompt="test prompt",
            seed=123,
            output=Path("output.png"),
        )
        self.assertIn("--auto-fit", args)
        self.assertIn("--diffusion-fa", args)
        self.assertIn("--max-vram", args)
        self.assertNotIn("--vae-on-cpu", args)
        self.assertNotIn("--taesd", args)
        self.assertEqual("50", args[args.index("--steps") + 1])
        self.assertEqual("123", args[args.index("--seed") + 1])
        self.assertEqual("-1.0", args[args.index("--max-vram") + 1])

    def test_prompt_is_one_argument(self):
        prompt = 'A sign reading "EXACT TEXT" beside a cat'
        args = lab.build_sd_arguments(
            self.manifest,
            executable=Path("sd-cli.exe"),
            bundle_name="qwen-2512-q6-quality",
            profile_name="smoke",
            prompt=prompt,
            seed=1,
            output=Path("output.png"),
        )
        self.assertEqual(prompt, args[args.index("-p") + 1])


class PngTests(unittest.TestCase):
    def test_png_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.png"
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + (1024).to_bytes(4, "big")
                + (768).to_bytes(4, "big")
            )
            self.assertEqual((1024, 768), lab.png_dimensions(path))

    def test_ocr_normalization(self):
        self.assertEqual(
            "TEA CAKE GOOD BOOKS",
            lab.normalize_ocr_text("Tea, Cake & Good Books!"),
        )

    def test_fuzzy_ocr_handles_common_character_confusion(self):
        score = lab.best_ocr_phrase_score(
            "TEA CAKE GOOD BOOKS",
            {"crop": "TEA, CAKE, G00D BOOKS"},
        )
        self.assertGreater(score, 0.85)


if __name__ == "__main__":
    unittest.main()
