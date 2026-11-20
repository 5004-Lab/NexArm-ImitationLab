"""Offline action prediction in physical units and trial summaries."""

import csv
import time
from pathlib import Path

import numpy as np

from .raw import write_json


def action_metrics(truth, predictions, latencies=None):
    truth = np.asarray(truth)
    predictions = np.asarray(predictions)
    error = predictions - truth
    result = {
        "per_axis_mae": np.abs(error).mean(axis=0).tolist(),
        "per_axis_rmse": np.sqrt((error**2).mean(axis=0)).tolist(),
        "joint_rmse_degree": float(np.sqrt((error[:, :5] ** 2).mean())),
        "gripper_mae_open_fraction": float(np.abs(error[:, -1]).mean()),
        "gripper_open_closed_accuracy": float(
            ((truth[:, -1] >= 0.5) == (predictions[:, -1] >= 0.5)).mean()
        ),
        "frames": len(truth),
    }
    if latencies is not None:
        result["latency_ms"] = {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        }
    return result


def load_policy(path, device):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(path).to(device)
    policy.config.device = device
    policy.eval()
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def evaluate(root, checkpoint, output, split="test", device="cpu"):
    import torch

    from .train import dataset, loader

    policy, pre, post = load_policy(checkpoint, device)
    ds = dataset(root, split)
    truth = []
    pred = []
    latency = []
    episodes = []
    last_episode = None
    with torch.no_grad():
        for item in loader(ds, 1, False):
            eid = int(item["episode_index"].item())
            if eid != last_episode:
                policy.reset()
                last_episode = eid
            start = time.perf_counter()
            predicted = post(policy.select_action(pre(item))).cpu().numpy()[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency.append((time.perf_counter() - start) * 1000)
            truth.append(item["action"].numpy().reshape(-1, 6)[0])
            pred.append(predicted)
            episodes.append(eid)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = action_metrics(truth, pred, latency)
    write_json(output / "metrics.json", metrics)
    np.savez(
        output / "predictions.npz",
        truth=truth,
        prediction=pred,
        episode_index=episodes,
        latency_ms=latency,
    )
    plot_actions(truth, pred, episodes, output / "actions.png")
    return metrics


def plot_actions(truth, pred, episodes, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .raw import AXES

    truth = np.asarray(truth)
    pred = np.asarray(pred)
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    boundaries = np.flatnonzero(np.diff(episodes)) + 1
    for j, ax in enumerate(axes):
        ax.plot(truth[:, j], label="Recorded")
        ax.plot(pred[:, j], label="Predicted", alpha=0.8)
        for x in boundaries:
            ax.axvline(x, color="gray", linestyle=":")
        ax.set_ylabel(AXES[j] + (" (deg)" if j < 5 else " (open)"))
    axes[0].legend()
    axes[-1].set_xlabel("Frame (dotted lines: episode boundaries)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def summarize_trials(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    success = sum(r["success"].lower() in ("true", "1", "yes") for r in rows)
    return {
        "trials": len(rows),
        "successes": success,
        "success_rate": success / len(rows) if rows else None,
        "mean_completion_seconds": float(np.mean([float(r["completion_seconds"]) for r in rows]))
        if rows
        else None,
        "interventions": sum(int(r["interventions"]) for r in rows),
        "failures": {
            k: sum(r["failure_type"] == k for r in rows)
            for k in sorted({r["failure_type"] for r in rows if r["failure_type"]})
        },
    }
