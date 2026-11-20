"""Check exact state/action alignment and episode boundaries after conversion."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_pipeline import fixture

from imitation_lab.convert import clips, convert


class ClipTests(unittest.TestCase):
    def test_timestamp_gaps_split_action_chunks(self):
        rows = [
            {
                "frame_index": i,
                "timestamp": i / 5 + (1 if i >= 3 else 0),
                "action": [0] * 6,
                "observation.state": [0] * 6,
            }
            for i in range(6)
        ]
        self.assertEqual(clips(rows, min_frames=1, fps=5), [[0, 1, 2], [3, 4, 5]])

    def test_excluded_segment_splits_clips(self):
        rows = [
            {"frame_index": i, "action": [0] * 6, "observation.state": [0] * 6} for i in range(5)
        ]
        self.assertEqual(clips(rows, min_frames=1, excluded={2}), [[0, 1], [3, 4]])

    def test_nan_action_is_not_a_training_target(self):
        rows = [{"frame_index": 0, "action": [float("nan")] * 6, "observation.state": [0] * 6}]
        with self.assertRaises(ValueError):
            clips(rows, min_frames=1)


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "LeRobot integration dependency")
class ConversionTests(unittest.TestCase):
    def test_round_trip_commands_and_padding(self):
        from imitation_lab.train import dataset

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [fixture(root / f"{i}.zip", episode_id=f"episode-{i}") for i in range(3)]
            info = convert(paths, root / "converted", image_size=(64, 64))
            self.assertFalse(set(info["splits"]["train"]) & set(info["splits"]["test"]))
            data = dataset(root / "converted", "train", 10)
            first = data[0]
            last = data[len(data) - 1]
            np.testing.assert_allclose(first["action"][0], [0, 0, 0, 0, 0, 0.5])
            np.testing.assert_allclose(first["action"][1], [0.1, 0, 0, 0, 0, 0.5])
            self.assertEqual(tuple(first["observation.images.third"].shape), (3, 64, 64))
            self.assertEqual(last["action_is_pad"].tolist(), [False] + [True] * 9)
