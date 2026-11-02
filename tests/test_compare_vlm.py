import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from scripts import compare_vlm
from scripts import report_vlm


class ComparisonTests(unittest.TestCase):
    def test_small_memory_float32_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "samples").mkdir()
            (root / "samples/manifest.json").write_text(json.dumps({"clips": [{"id": "a"}]}))
            for name in ("scripts/compare_vlm.py", "app/sampling.py", "app/core.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test fixture")
            with patch.object(compare_vlm, "ROOT", root), patch.object(compare_vlm, "memory_limit_bytes", return_value=8 * 1024**3), patch("sys.argv", ["compare_vlm.py", "--run-id", "limited", "--dtype", "float32"]):
                with self.assertRaises(MemoryError):
                    compare_vlm.main()
            record = json.loads((root / "experiments/limited/run.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["completed_inferences"], 0)
            self.assertFalse((root / "experiments/limited/summary.json").exists())

    def test_evidence_path_cannot_leave_run_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                report_vlm.safe_path(root, "../outside.png")

    def test_failed_run_cannot_generate_completed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.json").write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):
                report_vlm.build_report(root)
            self.assertFalse((root / "review.html").exists())

    def test_message_contains_only_time_pixels_and_fixed_instruction(self):
        image = Image.new("RGB", (32, 32))
        messages = compare_vlm.messages_for([(2.0, image)])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], [
            {"type": "text", "text": "Frame at 2.00s"},
            {"type": "image", "image": image},
            {"type": "text", "text": compare_vlm.INSTRUCTION},
        ])
        image.close()

    def test_requested_subset_is_exact_and_unknown_or_duplicate_ids_fail(self):
        clips = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(compare_vlm.select_clips(clips, "b,a"), [clips[1], clips[0]])
        for ids in ("a,a", "c", "a,"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                compare_vlm.select_clips(clips, ids)

    def test_existing_run_is_rejected_without_touching_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "samples").mkdir()
            (root / "samples/manifest.json").write_text(json.dumps({"clips": [{"id": "a"}]}))
            output = root / "experiments/existing"
            output.mkdir(parents=True)
            record = output / "run.json"
            record.write_text('{"status":"completed"}')
            with patch.object(compare_vlm, "ROOT", root), patch("sys.argv", ["compare_vlm.py", "--run-id", "existing"]):
                with self.assertRaises(FileExistsError):
                    compare_vlm.main()
            self.assertEqual(record.read_text(), '{"status":"completed"}')


if __name__ == "__main__":
    unittest.main()
