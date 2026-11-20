"""Serve complete ACT action chunks on the training GPU; never control hardware."""

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image


class ACTInference:
    def __init__(self, checkpoint, device):
        import torch

        from .evaluate import load_policy

        self.torch = torch
        self.policy, self.pre, self.post = load_policy(checkpoint, device)
        if self.policy.config.type != "act":
            raise ValueError("This endpoint requires an ACT checkpoint")
        self.lock = threading.Lock()
        self.info = {
            "model": "act",
            "checkpoint": str(checkpoint),
            "device": device,
            "action_units": "encoder_ticks",
            "images": {k: list(v.shape) for k, v in self.policy.config.image_features.items()},
            "chunk_size": self.policy.config.chunk_size,
        }
        from safetensors.torch import load_file

        stats = load_file(str(next(Path(checkpoint).glob("*normalizer_processor.safetensors"))))
        self.state_mean = stats["observation.state.mean"].numpy()
        self.state_std = stats["observation.state.std"].numpy()
        self.info["state_mean"] = self.state_mean.tolist()
        self.info["state_std"] = self.state_std.tolist()

    def predict(self, data):
        torch = self.torch
        start = time.monotonic()
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("state must contain six finite raw encoder positions")
        obs = {"observation.state": torch.from_numpy(state)}
        for name, shape in self.info["images"].items():
            with Image.open(io.BytesIO(base64.b64decode(data["images"][name]))) as image:
                rgb = np.asarray(image.convert("RGB").resize((shape[2], shape[1]))).copy()
            obs[name] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        with self.lock, torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(obs))
            # Process every six-dimensional action with the saved output normalizer.
            actions = self.post(chunk.flatten(0, 1)).cpu().numpy()
        if actions.shape[1] != 6 or not np.isfinite(actions).all():
            raise ValueError("ACT returned invalid native actions")
        return {
            "actions": actions.tolist(),
            "inference_ms": (time.monotonic() - start) * 1000,
            "state_zscore": ((state - self.state_mean) / np.maximum(self.state_std, 1e-8)).tolist(),
        }


def serve(checkpoint, device, host, port):
    inference = ACTInference(checkpoint, device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(inference.info)
            else:
                self.reply({"error": "unknown endpoint"}, 404)

        def do_POST(self):
            if self.path != "/predict":
                self.reply({"error": "unknown endpoint"}, 404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(inference.predict(data))
            except (ValueError, KeyError) as error:
                self.reply({"error": str(error)}, 400)

        def log_message(self, fmt, *args):
            if args and str(args[1]) != "200":
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"listening": f"http://{host}:{port}", **inference.info}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    serve(args.checkpoint, args.device, args.host, args.port)


if __name__ == "__main__":
    main()
