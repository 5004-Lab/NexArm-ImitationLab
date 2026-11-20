"""ACT training using LeRobot policy, dataset and normalization processors."""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from .models import make_policy, processors, validation_prediction
from .raw import write_json


def dataset(root, split, chunk_size=1, diffusion=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    info = json.loads((root / "dataset.json").read_text())
    delta = {"action": [i / info["fps"] for i in range(chunk_size)]}
    if diffusion:
        for name in json.loads((root / split / "meta/info.json").read_text())["features"]:
            if name.startswith("observation."):
                delta[name] = [0.0]
    return LeRobotDataset(
        repo_id=info["repo_id"] + "_" + split,
        root=root / split,
        delta_timestamps=delta,
        download_videos=False,
    )


def loader(ds, batch_size, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train(cfg, resume=None):
    root = Path(cfg["dataset"])
    if not (root / "dataset.json").exists():
        raise ValueError("convert episodes first: " + str(root))
    import torch

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; select cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    is_diffusion = cfg["policy"].get("type") == "diffusion"
    horizon = cfg["policy"].get("horizon", cfg["policy"].get("chunk_size", 1))
    ds = dataset(root, "train", horizon, is_diffusion)
    policy = make_policy(cfg, ds.meta)
    pre, post = processors(policy.config, ds.meta.stats)
    opt = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=cfg["policy"]["optimizer_lr"],
        weight_decay=cfg["policy"]["optimizer_weight_decay"],
    )
    tr = cfg["training"]
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=bool(resume))
    start_step = 1
    if resume:
        from .evaluate import load_policy

        policy, pre, post = load_policy(resume, cfg["device"])
        opt = torch.optim.AdamW(
            policy.get_optim_params(),
            lr=cfg["policy"]["optimizer_lr"],
            weight_decay=cfg["policy"]["optimizer_weight_decay"],
        )
        saved = torch.load(
            Path(resume).parent / "training_state.pt", map_location="cpu", weights_only=False
        )
        opt.load_state_dict(saved["optimizer"])
        start_step = saved["step"] + 1
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if torch.cuda.is_available() and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if tr["steps"] < start_step:
            raise ValueError("steps must exceed the checkpoint step")
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "dataset.json", json.loads((root / "dataset.json").read_text()))
    (out / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    val = dataset(root, "val", horizon, is_diffusion) if (root / "val").exists() else None
    batches = loader(ds, tr["batch_size"], True)
    iterator = iter(batches)
    log = (out / "loss.csv").open("a" if resume else "w", newline="")
    writer = csv.DictWriter(log, fieldnames=["step", "train_loss", "l1", "kl", "val_l1"])
    if not resume:
        writer.writeheader()
    for step in range(start_step, tr["steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        policy.train()
        loss, details = policy(pre(batch))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        if step % tr["log_freq"] == 0 or step == 1 or step == tr["steps"]:
            val_l1 = None
            if val:
                policy.eval()
                total = 0.0
                count = 0
                with torch.no_grad():
                    for j, vbatch in enumerate(loader(val, tr["batch_size"], False)):
                        if j >= tr["validation_batches"]:
                            break
                        vb = pre(vbatch)
                        prediction = validation_prediction(policy, vb)
                        target = vb["action"][:, 0]
                        mask = (~vb["action_is_pad"][:, 0]).unsqueeze(-1).expand_as(prediction)
                        total += (prediction - target).abs()[mask].sum().item()
                        count += mask.sum().item()
                val_l1 = total / count
            record = {
                "step": step,
                "train_loss": loss.item(),
                "l1": (details or {}).get("l1_loss"),
                "kl": (details or {}).get("kld_loss"),
                "val_l1": val_l1,
            }
            writer.writerow(record)
            log.flush()
            print(json.dumps(record), flush=True)
        if step % tr["save_freq"] == 0 or step == tr["steps"]:
            path = out / "checkpoints" / str(step) / "pretrained_model"
            policy.save_pretrained(path)
            pre.save_pretrained(path)
            post.save_pretrained(path)
            write_json(path / "train_config.json", cfg)
            write_json(path / "dataset_spec.json", json.loads((root / "dataset.json").read_text()))
            torch.save(
                {
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                path.parent / "training_state.pt",
            )
            last = out / "checkpoints" / "last"
            if last.is_symlink():
                last.unlink()
            last.symlink_to(str(step), target_is_directory=True)
    log.close()
    plot_losses(out / "loss.csv", out / "loss.png")
    return str(out / "checkpoints/last/pretrained_model")


def plot_losses(csv_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for field in ("train_loss", "val_l1"):
        points = [(int(r["step"]), float(r[field])) for r in rows if r[field]]
        if points:
            plt.plot(*zip(*points), label=field)
    plt.xlabel("Step")
    plt.ylabel("Loss (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
