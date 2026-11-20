"""Use the running NexArm console for observations and planned joint motions."""

import io
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


class Console:
    def __init__(self, url, joints):
        self.url = url.rstrip("/")
        self.joints = joints
        self.operator = str(uuid.uuid4())
        self.key = None

    def request(self, path, data=None):
        headers = {"X-Operator": self.operator}
        if data is not None:
            headers.update(
                {"Content-Type": "application/json", "Origin": self.url, "X-Control-Key": self.key}
            )
        req = urllib.request.Request(
            self.url + path,
            headers=headers,
            data=json.dumps(data).encode() if data is not None else None,
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read()

    def state(self):
        return json.loads(self.request("/api/state"))

    def vector(self, state):
        pulses = [item[0] for item in state["feedback"]]
        jaw = self.joints[-1]
        return np.array(
            state["angles"][:5]
            + [float(np.clip((jaw["max"] - pulses[-1]) / (jaw["max"] - jaw["min"]), 0, 1))],
            dtype=np.float32,
        )

    def image(self, camera, size):
        path = {
            "third": "/api/camera/external",
            "wrist": "/api/camera",
            "front": "/api/camera/realsense",
        }[camera]
        with Image.open(io.BytesIO(self.request(path))) as image:
            return np.asarray(image.convert("RGB").resize(size))

    def start(self):
        self.key = json.loads(self.request("/api/session"))["key"]
        self.command("arm")

    def command(self, name, data=None):
        return json.loads(self.request("/api/" + name, data or {}))

    def send(self, action):
        jaw = self.joints[-1]
        pulse = jaw["max"] - action[-1] * (jaw["max"] - jaw["min"])
        angle = math.degrees((pulse - jaw["zero"]) * jaw["sign"] / jaw["ticks"])
        self.command("move-joints", {"angles": list(map(float, action[:5])) + [angle]})


class MotionLimits:
    def __init__(self, joints, fps=5):
        self.dt = 1 / fps
        self.velocity = np.zeros(6)
        bounds = [
            sorted(
                math.degrees((j[p] - j["zero"]) * j["sign"] / j["ticks"]) for p in ("min", "max")
            )
            for j in joints[:5]
        ]
        self.low = np.array([b[0] for b in bounds] + [0])
        self.high = np.array([b[1] for b in bounds] + [1])
        self.speed = np.array([5] * 5 + [0.25])
        self.accel = np.array([10] * 5 + [0.5])

    def apply(self, action, state):
        action = np.asarray(action)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("model must output six finite values")
        desired = np.clip(action, self.low, self.high)
        velocity = np.clip((desired - state) / self.dt, -self.speed, self.speed)
        velocity = np.clip(
            velocity, self.velocity - self.accel * self.dt, self.velocity + self.accel * self.dt
        )
        self.velocity = velocity
        return np.clip(state + velocity * self.dt, self.low, self.high)


def rollout(
    checkpoint, url, joints_path, output, steps=100, device="cpu", execute=False, stop_file="STOP"
):
    import torch

    from .evaluate import load_policy

    policy, pre, post = load_policy(checkpoint, device)
    console = Console(url, json.loads(Path(joints_path).read_text()))
    limits = MotionLimits(console.joints)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if execute:
        console.start()
    try:
        with path.open("w") as log:
            for i in range(steps):
                if Path(stop_file).exists():
                    break
                start = time.monotonic()
                state = console.state()
                if not state["connected"]:
                    raise RuntimeError("robot feedback unavailable")
                vector = console.vector(state)
                observation = {"observation.state": torch.from_numpy(vector)}
                for name, feature in policy.config.image_features.items():
                    camera = name.rsplit(".", 1)[-1]
                    _, h, w = feature.shape
                    observation[name] = (
                        torch.from_numpy(console.image(camera, (w, h)).copy())
                        .permute(2, 0, 1)
                        .float()
                        / 255
                    )
                with torch.no_grad():
                    prediction = post(policy.select_action(pre(observation))).cpu().numpy()[0]
                action = limits.apply(prediction, vector)
                if execute:
                    console.command("heartbeat")
                    if not state["moving"]:
                        console.send(action)
                record = {
                    "step": i,
                    "mode": "execute" if execute else "shadow",
                    "state": vector.tolist(),
                    "prediction": prediction.tolist(),
                    "limited_action": action.tolist(),
                    "latency_ms": (time.monotonic() - start) * 1000,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                time.sleep(max(0, 0.2 - (time.monotonic() - start)))
    finally:
        if execute:
            console.command("stop")
