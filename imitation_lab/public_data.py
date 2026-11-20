"""Import a native LeRobot dataset without relabeling its action coordinates."""

import json
from pathlib import Path

import numpy as np

from .convert import split_ids
from .raw import write_json


def import_public(
    source,
    repo_id,
    output,
    cameras=("wrist",),
    fps=5,
    image_size=(96, 96),
    seed=42,
    max_episodes=None,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = Path(source)
    original = json.loads((source / "meta/info.json").read_text())
    if original["fps"] % fps:
        raise ValueError("choose an integer divisor of source fps; no frame duplication")
    stride = original["fps"] // fps
    ds = LeRobotDataset(repo_id=repo_id, root=source, download_videos=False, video_backend="pyav")
    ids = sorted(set(int(x) for x in ds.hf_dataset["episode_index"]))
    if max_episodes:
        ids = ids[:max_episodes]
    splits = split_ids(ids, seed)
    features = {k: dict(original["features"][k]) for k in ("observation.state", "action")}
    for feature in features.values():
        feature["shape"] = tuple(feature["shape"])
    for camera in cameras:
        features["observation.images." + camera] = {
            "dtype": "image",
            "shape": (image_size[1], image_size[0], 3),
            "names": ["height", "width", "channels"],
        }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = {
        "repo_id": "local/public_nexarm",
        "source_repo": repo_id,
        "robot_type": original["robot_type"],
        "fps": fps,
        "source_fps": original["fps"],
        "axes": features["action"]["names"],
        "units": ["source_native"] * features["action"]["shape"][0],
        "action_source": "source_dataset_action",
        "robot_profile": "source_native",
        "cameras": list(cameras),
        "image_size": list(image_size),
        "seed": seed,
        "splits": splits,
        "clips": {},
    }
    # Read each source video sequentially once. Avoid per-frame random AV1 seeks.
    import av
    import pandas as pd

    data = pd.concat(
        [pd.read_parquet(p) for p in sorted((source / "data").glob("**/*.parquet"))],
        ignore_index=True,
    )
    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((source / "meta/episodes").glob("**/*.parquet"))],
        ignore_index=True,
    ).set_index("episode_index")
    tasks = pd.read_parquet(source / "meta/tasks.parquet")
    task_column = "task" if "task" in tasks.columns else None
    images = {}
    for camera in cameras:
        key = "observation.images." + camera
        required = {}
        for eid in ids:
            em = episodes.loc[eid]
            video_key = (
                camera,
                int(em["videos/" + key + "/chunk_index"]),
                int(em["videos/" + key + "/file_index"]),
            )
            rows = data[data.episode_index == eid].sort_values("frame_index").iloc[::stride]
            indices = [
                round((float(em["videos/" + key + "/from_timestamp"]) + float(t)) * original["fps"])
                for t in rows.timestamp
            ]
            required.setdefault(video_key, set()).update(indices)
        for video in sorted((source / "videos" / key).glob("**/*.mp4")):
            video_key = (
                camera,
                int(video.parent.name.split("-")[-1]),
                int(video.stem.split("-")[-1]),
            )
            images[video_key] = {}
            with av.open(str(video)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in required.get(video_key, set()):
                        images[video_key][i] = np.asarray(frame.to_image().resize(image_size))
    for split, episode_ids in splits.items():
        if not episode_ids:
            continue
        dest = LeRobotDataset.create(
            repo_id=info["repo_id"] + "_" + split,
            root=output / split,
            fps=fps,
            features=features,
            robot_type=original["robot_type"],
            use_videos=False,
        )
        mappings = []
        try:
            for eid in episode_ids:
                rows = data[data.episode_index == eid].sort_values("frame_index").iloc[::stride]
                em = episodes.loc[eid]
                for row in rows.to_dict("records"):
                    frame = {
                        k: np.asarray(row[k], dtype=np.float32)
                        for k in ("observation.state", "action")
                    }
                    for camera in cameras:
                        key = "observation.images." + camera
                        chunk = int(em["videos/" + key + "/chunk_index"])
                        file = int(em["videos/" + key + "/file_index"])
                        offset = float(em["videos/" + key + "/from_timestamp"])
                        source_i = round((offset + float(row["timestamp"])) * original["fps"])
                        frame[key] = images[(camera, chunk, file)][source_i]
                    task_id = int(row["task_index"])
                    frame["task"] = (
                        str(tasks.loc[task_id, task_column])
                        if task_column
                        else str(tasks.index[task_id])
                    )
                    dest.add_frame(frame)
                dest.save_episode()
                mappings.append(
                    {
                        "source_episode": eid,
                        "source_frames": rows.frame_index.astype(int).tolist(),
                        "source_timestamps": rows.timestamp.tolist(),
                    }
                )
        finally:
            dest.finalize()
        info["clips"][split] = mappings
    write_json(output / "dataset.json", info)
    return info
