"""Command line entry points for data preparation and training."""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def training_config(path, resume=None):
    import yaml

    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    if resume:
        cfg = json.loads((Path(resume) / "train_config.json").read_text())
    else:
        cfg = yaml.safe_load(path.read_text())
    for key in ("dataset", "output"):
        value = Path(os.path.expandvars(os.path.expanduser(cfg[key])))
        cfg[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="nexarm-il", description="NexArm data, ACT training and evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="Show an episode structure")
    info.add_argument("episode")
    convert = commands.add_parser("convert", help="Convert episode ZIPs to LeRobot")
    convert.add_argument("episodes", nargs="+")
    convert.add_argument("--output", required=True)
    convert.add_argument(
        "--cameras", nargs="+", choices=["third", "wrist", "front"], default=["third"]
    )
    convert.add_argument("--seed", type=int, default=42)
    convert.add_argument("--repo-id", default="local/nexarm_screwdriver_v1")
    train = commands.add_parser("train", help="Train or resume ACT")
    train.add_argument("--config", default="configs/act_5hz_gpu.yaml")
    train.add_argument("--dataset")
    train.add_argument("--device")
    train.add_argument("--steps", type=int)
    train.add_argument("--output")
    train.add_argument("--resume", help="pretrained_model directory of a saved training checkpoint")
    offline = commands.add_parser("evaluate", help="Evaluate a checkpoint offline")
    offline.add_argument("--dataset", required=True)
    offline.add_argument("--checkpoint", required=True)
    offline.add_argument("--output", required=True)
    offline.add_argument("--device", default="cuda")
    offline.add_argument("--split", choices=["train", "val", "test"], default="test")
    rollout = commands.add_parser("rollout", help="Shadow inference or low-speed robot execution")
    rollout.add_argument("--checkpoint", required=True)
    rollout.add_argument("--url", default="http://127.0.0.1:8765")
    rollout.add_argument("--joints", default="/home/ztl/nexarm-console/src/joints.json")
    rollout.add_argument("--output", default="outputs/shadow.jsonl")
    rollout.add_argument("--steps", type=int, default=100)
    rollout.add_argument("--device", default="cpu")
    rollout.add_argument("--execute", action="store_true")
    rollout.add_argument("--stop-file", default="STOP")
    trials = commands.add_parser("trials", help="Summarize real robot trials")
    trials.add_argument("csv")
    public = commands.add_parser("import-public", help="Import a local native LeRobot dataset")
    public.add_argument("--source", required=True)
    public.add_argument("--repo-id", required=True)
    public.add_argument("--output", required=True)
    public.add_argument("--fps", type=int, default=5)
    public.add_argument("--cameras", nargs="+", default=["wrist"])
    public.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-public":
            from .public_data import import_public

            result = import_public(
                args.source,
                args.repo_id,
                args.output,
                cameras=args.cameras,
                fps=args.fps,
                seed=args.seed,
            )["splits"]
        elif args.command == "info":
            from .raw import Episode, describe

            with Episode(args.episode) as episode:
                result = describe(episode)
        elif args.command == "convert":
            from .convert import convert

            result = convert(
                args.episodes,
                args.output,
                repo_id=args.repo_id,
                cameras=args.cameras,
                seed=args.seed,
            )["splits"]
        elif args.command == "train":
            from .train import train

            cfg = training_config(args.config, args.resume)
            for key in ("device", "output", "dataset"):
                if getattr(args, key) is not None:
                    cfg[key] = getattr(args, key)
            if args.steps is not None:
                cfg["training"]["steps"] = args.steps
            result = train(cfg, resume=args.resume)
        elif args.command == "evaluate":
            from .evaluate import evaluate

            result = evaluate(args.dataset, args.checkpoint, args.output, args.split, args.device)
        elif args.command == "rollout":
            from .robot import rollout

            result = rollout(
                args.checkpoint,
                args.url,
                args.joints,
                args.output,
                args.steps,
                args.device,
                args.execute,
                args.stop_file,
            )
        else:
            from .evaluate import summarize_trials

            result = summarize_trials(args.csv)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, str(error) + "\n")
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
