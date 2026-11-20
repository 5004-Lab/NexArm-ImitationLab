"""Reader for NexArm's JSONL and JPEG episode export."""

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

AXES = ["J1", "J2", "J3", "J4", "J5", "gripper"]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Episode:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = ZipFile(path)
        self.manifest = json.loads(self.archive.read("manifest.json"))
        self.rows = self.jsonl("frames.jsonl")
        self.segments = self.jsonl("segments.jsonl")

    def jsonl(self, name):
        return [json.loads(line) for line in self.archive.read(name).splitlines() if line.strip()]

    def image(self, row, camera):
        with Image.open(
            io.BytesIO(self.archive.read(row["observation.images." + camera]))
        ) as image:
            return np.asarray(image.convert("RGB"))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()


def describe(ep):
    m = ep.manifest
    return {
        "episode": m["id"],
        "frames": len(ep.rows),
        "fps": m["fps"],
        "axes": m["axes"],
        "units": m["units"],
        "result": m["result"],
        "actions_present": sum(r.get("action") is not None for r in ep.rows),
        "task": m.get("metadata", {}).get("task", ""),
        "cameras": [
            k.rsplit(".", 1)[-1] for k in ep.rows[0] if k.startswith("observation.images.")
        ],
        "duration_seconds": ep.rows[-1]["timestamp"] - ep.rows[0]["timestamp"],
    }
