"""Convert real NexArm command/state pairs to LeRobot v3 datasets."""

import random
from pathlib import Path

import numpy as np
from PIL import Image

from .raw import Episode, write_json


def split_ids(ids, seed=42):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    if len(ids) < 3:
        return {"train": ids, "val": [], "test": []}
    nv = max(1, round(len(ids) * 0.15))
    nt = max(1, round(len(ids) * 0.15))
    return {"train": ids[: -nv - nt], "val": ids[-nv - nt : -nt], "test": ids[-nt:]}


def clips(rows, min_frames=10):
    runs = []
    run = []
    for i, row in enumerate(rows):
        if row.get("action") is None or row.get("observation.state") is None:
            if len(run) >= min_frames:
                runs.append(run)
            run = []
            continue
        if run and row["frame_index"] != rows[run[-1]]["frame_index"] + 1:
            if len(run) >= min_frames:
                runs.append(run)
            run = []
        run.append(i)
    if len(run) >= min_frames:
        runs.append(run)
    return runs


def convert(
    paths,
    output,
    repo_id="local/nexarm_screwdriver_v1",
    cameras=("third",),
    seed=42,
    image_size=(320, 240),
    min_frames=10,
):
    sources = {}
    metadata = {}
    runs = {}
    for path in paths:
        with Episode(path) as ep:
            eid = ep.manifest["id"]
            if eid in sources:
                raise ValueError("same episode supplied twice: " + eid)
            if ep.manifest["axes"] != ["J1", "J2", "J3", "J4", "J5", "gripper"]:
                raise ValueError("this NexArm uses J1..J5 and gripper")
            runs[eid] = clips(ep.rows, min_frames)
            if not runs[eid]:
                raise ValueError(
                    f"{eid}: no usable action/state clips; this export has no command labels"
                )
            sources[eid] = Path(path)
            metadata[eid] = ep.manifest
    fps = next(iter(metadata.values()))["fps"]
    if any(m["fps"] != fps for m in metadata.values()):
        raise ValueError("use one sampling frequency per dataset")
    splits = split_ids(list(sources), seed)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        k: {"dtype": "float32", "shape": (6,), "names": next(iter(metadata.values()))["axes"]}
        for k in ("observation.state", "action")
    }
    for camera in cameras:
        features["observation.images." + camera] = {
            "dtype": "image",
            "shape": (image_size[1], image_size[0], 3),
            "names": ["height", "width", "channels"],
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = {
        "repo_id": repo_id,
        "fps": fps,
        "cameras": list(cameras),
        "image_size": list(image_size),
        "axes": next(iter(metadata.values()))["axes"],
        "units": next(iter(metadata.values()))["units"],
        "seed": seed,
        "splits": splits,
        "episodes": metadata,
        "clips": {},
        "action_source": "recorded_command",
    }
    for split, ids in splits.items():
        if not ids:
            continue
        ds = LeRobotDataset.create(
            repo_id=repo_id + "_" + split,
            root=output / split,
            fps=fps,
            robot_type="nexarm",
            features=features,
            use_videos=False,
        )
        mapping = []
        try:
            for eid in ids:
                with Episode(sources[eid]) as ep:
                    for run in runs[eid]:
                        mapping.append(
                            {
                                "source_episode": eid,
                                "source_frames": [ep.rows[i]["frame_index"] for i in run],
                            }
                        )
                        for i in run:
                            row = ep.rows[i]
                            frame = {
                                k: np.asarray(row[k], dtype=np.float32)
                                for k in ("action", "observation.state")
                            }
                            for camera in cameras:
                                frame["observation.images." + camera] = np.asarray(
                                    Image.fromarray(ep.image(row, camera)).resize(image_size)
                                )
                            frame["task"] = row.get("task") or ep.manifest["metadata"]["task"]
                            ds.add_frame(frame)
                        ds.save_episode()
        finally:
            ds.finalize()
        info["clips"][split] = mapping
    write_json(output / "dataset.json", info)
    return info
