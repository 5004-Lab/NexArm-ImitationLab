"""Live shadow prediction snapshots for a read-only static preview page."""

import json
from pathlib import Path

from PIL import Image


class LivePreview:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        template = Path(__file__).with_name("preview.html").read_text()
        (self.directory / "index.html").write_text(template)

    def update(self, record, images, third):
        for name, array in {**images, "third": third}.items():
            target = self.directory / (name + ".jpg")
            temp = target.with_suffix(".tmp")
            Image.fromarray(array).save(temp, format="JPEG")
            temp.replace(target)
        temp = self.directory / "latest.tmp"
        temp.write_text(json.dumps(record, ensure_ascii=False))
        temp.replace(self.directory / "latest.json")
