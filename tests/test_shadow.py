"""Native public checkpoints must observe encoder values in shadow mode."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch runtime dependency")
class ShadowTests(unittest.TestCase):
    def test_native_shadow_uses_encoder_feedback_without_motion(self):
        import torch

        from imitation_lab.robot import rollout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset_spec.json").write_text(json.dumps({"robot_profile": "source_native"}))
            joints = [{"zero": 1000, "sign": 1, "ticks": 1000, "min": 0, "max": 4095}] * 6
            (root / "joints.json").write_text(json.dumps(joints))
            console = Mock()
            console.joints = joints
            console.state.return_value = {"connected": True, "feedback": [[1500]] * 6}
            console.vector.return_value = np.zeros(6, dtype=np.float32)
            policy = Mock()
            policy.config = SimpleNamespace(image_features={}, type="bc_mlp")
            policy.select_action.return_value = torch.tensor([[1501.0] * 6])
            with (
                patch("imitation_lab.robot.Console", return_value=console),
                patch(
                    "imitation_lab.evaluate.load_policy",
                    return_value=(policy, lambda b: b, lambda x: x),
                ),
            ):
                rollout(
                    root,
                    "http://unused",
                    root / "joints.json",
                    root / "prediction.jsonl",
                    steps=1,
                    stop_file=root / "STOP",
                )
            observed = policy.select_action.call_args.args[0]["observation.state"]
            self.assertEqual(observed.tolist(), [1500.0] * 6)
            console.start.assert_not_called()
            console.send.assert_not_called()
            console.command.assert_not_called()
            row = json.loads((root / "prediction.jsonl").read_text())
            self.assertEqual(row["prediction_native"], [1501.0] * 6)
