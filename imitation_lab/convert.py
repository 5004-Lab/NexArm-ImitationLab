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
    ntrain = max(1, min(len(ids) - 2, int(len(ids) * 0.70)))
    nv = max(1, int(len(ids) * 0.15))
    return {"train": ids[:ntrain], "val": ids[ntrain : ntrain + nv], "test": ids[ntrain + nv :]}


def clips(rows, min_frames=10, fps=None, excluded=()):
    runs = []
    run = []
    for i, row in enumerate(rows):
        if row["frame_index"] in excluded:
            if len(run) >= min_frames:
                runs.append(run)
            run = []
            continue
        if row.get("action") is None or row.get("observation.state") is None:
            if len(run) >= min_frames:
                runs.append(run)
            run = []
            continue
        for key in ("observation.state", "action"):
            values = np.asarray(row[key], dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"frame {row['frame_index']}: invalid {key}")
        if run and (
            row["frame_index"] != rows[run[-1]]["frame_index"] + 1
            or (fps and abs(row["timestamp"] - rows[run[-1]]["timestamp"] - 1 / fps) > 0.001)
        ):
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
    max_age_ms=100,
    max_span_ms=45,
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
            if ep.manifest["units"] != ["degree"] * 5 + ["open_fraction"]:
                raise ValueError("console export units must be degree / open_fraction")
            excluded = {
                i
                for segment in ep.segments
                if segment["kind"] == "exclude"
                for i in range(segment["first"], segment["last"] + 1)
            }
            for row in ep.rows:
                offsets = row.get("sync_ms")
                if offsets is not None:
                    keys = ["feedback", *cameras]
                    if any(k not in offsets for k in keys):
                        excluded.add(row["frame_index"])
                        continue
                    values = [offsets[k] for k in keys]
                    if (
                        any(not np.isfinite(v) or v > 0.01 or abs(v) > max_age_ms for v in values)
                        or max(values) - min(values) > max_span_ms
                        or set(keys) & set(row.get("repeated", []))
                    ):
                        excluded.add(row["frame_index"])
            runs[eid] = clips(ep.rows, min_frames, ep.manifest["fps"], excluded)
            if not runs[eid]:
                raise ValueError(
                    f"{eid}: no usable action/state clips; this export has no command labels"
                )
            if any(
                not 0 <= ep.rows[i][k][-1] <= 1
                for run in runs[eid]
                for i in run
                for k in ("action", "observation.state")
            ):
                raise ValueError("gripper must use open_fraction in [0,1]")
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
        "robot_profile": "nexarm_console",
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
                                "source_timestamps": [ep.rows[i]["timestamp"] for i in run],
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
