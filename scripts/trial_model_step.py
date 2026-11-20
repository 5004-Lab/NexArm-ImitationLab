"""One collision-checked native model proposal, capped at two degrees total.

This is an actuator-interface trial, not a complete autonomous grasp rollout.
Run on the robot server; it never disables the console planner.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from imitation_lab.evaluate import load_policy
from imitation_lab.robot import Console


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="data/models/public_pick/bc_mlp")
    p.add_argument("--console-root", default="/home/ztl/nexarm-console")
    p.add_argument("--preview-dir", default="outputs/live_preview")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, args.console_root)
    from nexarm import kinematics as k

    torch.set_num_threads(4)
    policy, pre, post = load_policy(args.checkpoint, "cpu")
    spec = json.loads((Path(args.checkpoint) / "dataset_spec.json").read_text())
    expected = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    if spec["axes"] != expected:
        raise ValueError("native axis order does not match NexArm servo IDs 1..6")
    console = Console("http://127.0.0.1:8765", k.JOINTS)
    s = console.state()
    start = np.array([v[0] for v in s["feedback"]], dtype=float)
    obs = {"observation.state": torch.tensor(start, dtype=torch.float32)}
    for name, f in policy.config.image_features.items():
        camera = name.rsplit(".", 1)[-1]
        _, h, w = f.shape
        obs[name] = (
            torch.from_numpy(console.image(camera, (w, h)).copy()).permute(2, 0, 1).float() / 255
        )
    with torch.no_grad():
        pred = post(policy.select_action(pre(obs))).cpu().numpy()[0]
    delta = pred - start
    angular = np.abs(delta[:5]) * np.array([180 / math.pi / j["ticks"] for j in k.JOINTS[:5]])
    ratio = min(1.0, 1.9 / max(float(angular.max()), 1e-6), 32.0 / max(abs(float(delta[-1])), 1e-6))
    final = np.round(start + delta * ratio).astype(int)
    waypoints = [
        np.round(start + (final - start) * i / 4).astype(int).tolist() for i in range(1, 5)
    ]
    previous = start.tolist()
    for target in waypoints:
        k.check_path(previous, target, k.joint_limits(previous))
        previous = target
    preview = Path(args.preview_dir)
    preview.mkdir(parents=True, exist_ok=True)
    record = {
        "mode": "trial_planned",
        "start_encoder": start.tolist(),
        "model_target_encoder": pred.tolist(),
        "bounded_target_encoder": final.tolist(),
        "max_joint_angle_degree": float(
            np.max(
                np.abs(final[:5] - start[:5])
                * np.array([180 / math.pi / j["ticks"] for j in k.JOINTS[:5]])
            )
        ),
        "timestamp": time.time(),
    }

    def status(message):
        record["message"] = message
        record["timestamp"] = time.time()
        temp = preview / "execution.tmp"
        temp.write_text(json.dumps(record))
        temp.replace(preview / "execution.json")
        print(message, flush=True)

    status("模型短段目标已通过路径检查；最多 2 度，不是完整抓取")
    if not args.execute:
        return
    for seconds in range(10, 0, -1):
        status(f"{seconds} 秒后开始最多 2 度的低速模型动作")
        time.sleep(1)
    current = console.state()
    if max(abs(v[0] - old) for v, old in zip(current["feedback"], start)) > 5:
        status("起始姿态在倒计时期间改变，本次未启动动作")
        return
    armed = False
    try:
        console.start()
        armed = True
        record["mode"] = "executing_trial"
        for index, target in enumerate(waypoints):
            if (preview / "TRIAL_STOP").exists():
                break
            actual = console.state()
            if not actual["connected"]:
                raise RuntimeError("feedback unavailable")
            measured = [v[0] for v in actual["feedback"]]
            k.check_path(measured, target, k.joint_limits(measured))
            angles = [
                math.degrees((pulse - j["zero"]) * j["sign"] / j["ticks"])
                for pulse, j in zip(target, k.JOINTS)
            ]
            status(f"正在执行第 {index + 1}/4 个小步")
            console.command("heartbeat")
            console.command("move-joints", {"angles": angles})
            deadline = time.monotonic() + 5
            while True:
                time.sleep(0.1)
                console.command("heartbeat")
                actual = console.state()
                if (preview / "TRIAL_STOP").exists():
                    raise RuntimeError("trial stopped by operator")
                if not actual["moving"]:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("motion did not finish within five seconds")
            time.sleep(0.6)
        record["end_encoder"] = [v[0] for v in console.state()["feedback"]]
        record["mode"] = "trial_complete"
        status("短段动作完成，已停止；本次未执行完整抓取")
    except Exception as error:
        record["mode"] = "trial_stopped"
        try:
            record["controller_result"] = console.state().get("result")
        except OSError:
            pass
        record["error"] = str(error)
        if hasattr(error, "read"):
            record["controller_response"] = error.read().decode()
        status("控制台中断了试动作，已停止；未完成抓取")
        raise
    finally:
        if armed:
            console.command("stop")
        record["timestamp"] = time.time()
        (preview / "execution-result.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
