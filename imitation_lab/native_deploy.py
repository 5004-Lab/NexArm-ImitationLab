"""ACT closed-loop deployment through the existing NexArm console planner."""

import base64
import io
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .preview import LivePreview
from .robot import Console

AXES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERAS = {
    "third": "/api/camera/external",
    "wrist": "/api/camera",
    "front": "/api/camera/realsense",
}


def request_json(url, data=None):
    request = urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Policy HTTP {error.code}: {error.read().decode()}") from error


def encoder_angles(pulses, joints):
    """Native follower ticks -> console degrees, including the sixth servo."""
    return [math.degrees((p - j["zero"]) * j["sign"] / j["ticks"]) for p, j in zip(pulses, joints)]


def bounded_target(prediction, actual, origin, joints, limits):
    """Keep native target direction with a common per-command step fraction."""
    actual, origin = np.asarray(actual), np.asarray(origin)
    degree_ticks = np.array([j["ticks"] * math.pi / 180 for j in joints[:5]])
    travel = np.r_[degree_ticks * limits["travel_deg"], limits["gripper_travel_ticks"]]
    step = np.floor(np.r_[degree_ticks * limits["step_deg"], limits["gripper_step_ticks"]])
    # An arm already outside configured soft bounds may return gradually.
    low = np.maximum(np.minimum([j["min"] for j in joints], actual), origin - travel)
    high = np.minimum(np.maximum([j["max"] for j in joints], actual), origin + travel)
    desired = np.clip(prediction, np.ceil(low), np.floor(high))
    delta = desired - actual
    ratio = min(1.0, 1.0 / max(1.0, float(np.max(np.abs(delta) / step))))
    return np.rint(actual + delta * ratio).astype(int).tolist()


class ControlLease:
    """Renew ownership independently of GPU requests and motion completion."""

    def __init__(self, console, interval=0.25):
        self.console = console
        self.interval = interval
        self.closed = threading.Event()
        self.error = None
        self.count = 0
        self.thread = threading.Thread(target=self.renew, daemon=True)

    def start(self):
        self.console.start()
        self.thread.start()

    def renew(self):
        while not self.closed.is_set():
            try:
                self.console.command("heartbeat")
                self.count += 1
            except Exception as error:
                self.error = error
                return
            self.closed.wait(self.interval)

    def check(self):
        if self.error:
            raise RuntimeError(f"Control heartbeat failed: {self.error}") from self.error

    def close(self):
        self.closed.set()
        self.thread.join(timeout=6)
        self.console.command("stop")


def checked_state(console, lease):
    state = console.state()
    if not state["connected"] or not state["armed"]:
        raise RuntimeError(state["result"] or "Robot control or feedback unavailable")
    lease.check()
    return state


def wait_motion(console, lease, stop_file, timeout=8):
    deadline = time.monotonic() + timeout
    while True:
        state = checked_state(console, lease)
        if Path(stop_file).exists():
            raise RuntimeError("Stopped by STOP file")
        if not state["moving"]:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Motion timeout: {state['result']}")
        time.sleep(0.05)


def deploy(config_path, output, cycles=8, execute=False, stop_file="STOP"):
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    sys.path.insert(0, config["console_root"])
    from nexarm import kinematics

    joints = kinematics.JOINTS
    console = Console(config["console_url"], joints)
    info = request_json(config["policy_url"] + "/health")
    if info["action_units"] != "encoder_ticks" or config["axes"] != AXES:
        raise ValueError("Deployment profile must use NexArm follower IDs 1..6 in raw ticks")
    if set(info["images"]) != set(config["cameras"]):
        raise ValueError("Camera map does not match checkpoint input names")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    preview = LivePreview(config["preview_dir"])
    initial = console.state()
    if not initial["connected"] or not initial["power_ok"]:
        raise ValueError(
            "Console joint feedback or power reading unavailable; restore the controller connection first"
        )
    origin = [v[0] for v in initial["feedback"]]
    lease = ControlLease(console)
    summary = {
        "mode": "execute" if execute else "shadow",
        "completed_cycles": 0,
        "commands_sent": 0,
        "start_encoder": origin,
        "model": info,
        "limits": config["limits"],
        "status": "running",
        "grasp_evaluated": False,
    }

    def status(mode, message):
        data = {"mode": mode, "message": message, "timestamp": time.time()}
        temp = Path(config["preview_dir"]) / "execution.tmp"
        temp.write_text(json.dumps(data, ensure_ascii=False))
        temp.replace(temp.with_name("execution.json"))
        print(message, flush=True)

    if Path(stop_file).exists():
        raise ValueError(f"Stop file exists: {stop_file}")
    from .trial_video import TrialVideo

    video = TrialVideo(console, directory) if execute else None
    started = False
    try:
        if execute:
            for second in range(5, 0, -1):
                status("countdown", f"ACT 有限行程联调：{second} 秒后启动")
                time.sleep(1)
                if Path(stop_file).exists():
                    raise RuntimeError("Stopped during countdown")
            current = console.state()
            if max(abs(v[0] - p) for v, p in zip(current["feedback"], origin)) > 5:
                raise RuntimeError("Starting posture changed during countdown")
            lease.start()
            started = True
            video.start()
        with (directory / "steps.jsonl").open("w") as log:
            for cycle in range(cycles):
                if Path(stop_file).exists():
                    raise RuntimeError("Stopped by STOP file")
                started_at = time.monotonic()
                state = checked_state(console, lease) if execute else console.state()
                actual = [v[0] for v in state["feedback"]]
                images, encoded = {}, {}
                for feature, source in config["cameras"].items():
                    content = console.request(CAMERAS[source])
                    encoded[feature] = base64.b64encode(content).decode()
                    with Image.open(io.BytesIO(content)) as image:
                        images[feature.rsplit(".", 1)[-1]] = np.asarray(image.convert("RGB"))
                result = request_json(
                    config["policy_url"] + "/predict", {"state": actual, "images": encoded}
                )
                prediction = result["actions"][0]
                state = checked_state(console, lease) if execute else console.state()
                measured = [v[0] for v in state["feedback"]]
                target = bounded_target(prediction, measured, origin, joints, config["limits"])
                record = {
                    "step": cycle,
                    "mode": summary["mode"],
                    "model": "act",
                    "state_native": actual,
                    "feedback_before_send": measured,
                    "prediction_native": prediction,
                    "target_encoder": target,
                    "timestamp": time.time(),
                    "prediction_units": "encoder_ticks",
                    "inference_ms": result["inference_ms"],
                    "state_zscore": result["state_zscore"],
                    "latency_ms": (time.monotonic() - started_at) * 1000,
                }
                try:
                    kinematics.check_path(measured, target, kinematics.joint_limits(measured))
                    record["path_valid"] = True
                except ValueError as error:
                    record.update(path_valid=False, path_error=str(error))
                if execute and not record["path_valid"]:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    raise RuntimeError(record["path_error"])
                if execute:
                    status(
                        "executing_trial",
                        f"ACT 连续联调 {cycle + 1}/{cycles}：读取图像 → 推理 → 路径检查 → 执行",
                    )
                    log.write(json.dumps({**record, "event": "command"}) + "\n")
                    log.flush()
                    console.command("policy-step", {"angles": encoder_angles(target, joints)})
                    summary["commands_sent"] += 1
                    time.sleep(max(0.10, 1 / config["fps"] - (time.monotonic() - started_at)))
                    state = checked_state(console, lease)
                    record["feedback_after"] = [v[0] for v in state["feedback"]]
                    record["controller_result"] = state["result"]
                    record["tracking_error_ticks"] = (
                        np.asarray(record["feedback_after"]) - target
                    ).tolist()
                else:
                    status(
                        "shadow",
                        f"ACT 无运动联调 {cycle + 1}/{cycles} · GPU {result['inference_ms']:.0f} ms",
                    )
                preview.update(record, images, images["front"])
                log.write(json.dumps(record) + "\n")
                log.flush()
                summary["completed_cycles"] += 1
                time.sleep(max(0, 1 / config["fps"] - (time.monotonic() - started_at)))
        summary["status"] = "completed"
        status("trial_complete" if execute else "shadow_complete", "ACT 联调完成；未评估抓取成功率")
    except Exception as error:
        summary["status"] = "stopped"
        summary["error"] = str(error)
        status("trial_stopped", f"ACT 联调已停止：{error}")
        raise
    finally:
        # Preserve the cause before stop() replaces the controller's result text.
        try:
            final = console.state()
            summary["controller_result"] = final["result"]
            summary["end_encoder"] = [v[0] for v in final["feedback"]]
        finally:
            summary["heartbeat_count"] = lease.count
            try:
                if started:
                    lease.close()
            finally:
                if video and started:
                    summary["video"] = video.close()
                (directory / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2)
                )
    return summary
