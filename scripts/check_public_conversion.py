"""Compare imported arrays and RGB pixels against the downloaded source."""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from imitation_lab.train import dataset

source = Path("data/public/nexarm_pick_20ep")
root = Path("data/processed/public_pick_5hz")
info = json.loads((root / "dataset.json").read_text())
data = pd.read_parquet(source / "data/chunk-000/file-000.parquet")
meta = pd.read_parquet(source / "meta/episodes/chunk-000/file-000.parquet").set_index(
    "episode_index"
)
checks = {}
for split in ("train", "val", "test"):
    ds = dataset(root, split)
    mapping = info["clips"][split][0]
    eid = mapping["source_episode"]
    source_frame = mapping["source_frames"][7]
    original = data[(data.episode_index == eid) & (data.frame_index == source_frame)].iloc[0]
    item = ds[7]
    np.testing.assert_array_equal(item["observation.state"].numpy(), original["observation.state"])
    np.testing.assert_array_equal(item["action"][0].numpy(), original["action"])
    tick = round(
        (meta.loc[eid]["videos/observation.images.wrist/from_timestamp"] + original.timestamp) * 30
    )
    with av.open(
        str(source / "videos/observation.images.wrist/chunk-000/file-000.mp4")
    ) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == tick:
                expected = np.asarray(frame.to_image().resize((96, 96)))
                actual = (
                    (item["observation.images.wrist"].numpy().transpose(1, 2, 0) * 255)
                    .round()
                    .astype("uint8")
                )
                np.testing.assert_array_equal(actual, expected)
                break
    checks[split] = {
        "source_episode": int(eid),
        "source_frame": int(source_frame),
        "source_video_frame": tick,
        "frames": len(ds),
    }
print(json.dumps(checks, indent=2))
print("SOURCE ACTION / STATE / RGB PIXELS: PASS")
