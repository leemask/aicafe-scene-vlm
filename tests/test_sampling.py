"""Sampling contracts and a decoding check against one bundled video."""
import json
import math
import unittest
from pathlib import Path

from app.sampling import hybrid_times, sample_frames, uniform_times

ROOT = Path(__file__).resolve().parents[1]


class SamplingTests(unittest.TestCase):
    def test_motion_adds_short_interval_while_preserving_uniform_anchors(self):
        anchors = uniform_times(0, 20, 3)
        selected = hybrid_times(0, 20, 6, [
            {"time": 12., "score": .8},
            {"time": 12.5, "score": .7},
            {"time": 5., "score": .1},
        ])
        self.assertEqual(len(selected), 6)
        self.assertTrue(set(anchors).issubset(selected))
        self.assertFalse(any(11.3 <= t <= 14.2 for t in anchors))
        self.assertTrue(any(11.3 <= t <= 14.2 for t in selected))
        self.assertEqual(selected, sorted(set(selected)))

    def test_quiet_scene_still_has_full_temporal_budget(self):
        for probes in ([], [{"time": 0., "score": 0.}]):
            with self.subTest(probes=probes):
                times = hybrid_times(0, 20, 6, probes)
                self.assertEqual(len(times), 6)
                self.assertEqual(times, sorted(set(times)))
                self.assertTrue(all(0 <= t < 20 for t in times))

    def test_invalid_intervals_and_budgets_rejected(self):
        for start, end, count in (
            (0, 0, 3), (-1, 20, 3), (0, math.inf, 3),
            (0, math.nan, 3), (0, 20, 0), (0, 20, 17),
            (0, 20, 2.5), (0, 20, True),
        ):
            with self.subTest(start=start, end=end, count=count):
                with self.assertRaises(ValueError):
                    uniform_times(start, end, count)

    def test_invalid_strategy_rejected_before_decoding(self):
        with self.assertRaises(ValueError):
            sample_frames("not-a-video", 0, 20, strategy="unknown")

    def test_bundled_video_returns_unique_actual_timestamps(self):
        manifest = json.loads((ROOT / "samples/manifest.json").read_text(encoding="utf-8"))
        clip = manifest["clips"][0]
        for strategy in ("uniform", "hybrid"):
            with self.subTest(strategy=strategy):
                frames = sample_frames(ROOT / clip["file"], 0, float(clip["duration"]), 6, strategy)
                times = [t for t, _ in frames]
                self.assertGreater(len(times), 0)
                self.assertLessEqual(len(times), 6)
                self.assertEqual(times, sorted(set(times)))
                self.assertTrue(all(0 <= t < clip["duration"] for t in times))
                for _, image in frames:
                    image.close()


if __name__ == "__main__":
    unittest.main()
