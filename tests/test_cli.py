import tempfile
import unittest
from pathlib import Path

from imitation_lab.cli import training_config


class ConfigTests(unittest.TestCase):
    def test_relative_paths_resolve_to_project(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "train.yaml"
            config.write_text("dataset: data/processed/demo\noutput: outputs/test\n")
            parsed = training_config(config)
            self.assertTrue(Path(parsed["dataset"]).is_absolute())
            self.assertTrue(parsed["dataset"].endswith("data/processed/demo"))

    def test_environment_expansion(self):
        import os
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "train.yaml"
            config.write_text("dataset: ${TEST_NEXARM_ROOT}/demo\noutput: outputs/test\n")
            with patch.dict(os.environ, {"TEST_NEXARM_ROOT": directory}):
                self.assertEqual(training_config(config)["dataset"], str(Path(directory) / "demo"))
