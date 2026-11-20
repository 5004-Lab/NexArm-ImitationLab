"""Train and evaluate all policies on the same fixed dataset split."""

import argparse
import json
from pathlib import Path

from imitation_lab.cli import training_config
from imitation_lab.evaluate import evaluate
from imitation_lab.raw import write_json
from imitation_lab.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", default="outputs/public_pick")
    parser.add_argument("--models", nargs="+", default=["act", "bc_mlp", "diffusion"])
    args = parser.parse_args()
    results = {}
    for kind in args.models:
        cfg = training_config(f"configs/policy/{kind}.yaml")
        cfg["training"]["steps"] = args.steps
        cfg["output"] = str(Path(args.output).resolve() / kind)
        print("TRAIN", kind, flush=True)
        checkpoint = train(cfg)
        result = evaluate(
            cfg["dataset"], checkpoint, Path(cfg["output"]) / "test", device=cfg["device"]
        )
        results[kind] = result
        write_json(Path(args.output) / "comparison.json", results)
        print(kind, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
