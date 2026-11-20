"""Record the real external camera during a bounded actuator trial."""

import json
import subprocess
import threading
import time


class TrialVideo:
    def __init__(self, console, directory, fps=5):
        self.console = console
        self.directory = directory
        self.fps = fps
        self.frames = 0
        self.error = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.capture, daemon=True)

    def start(self):
        self.thread.start()

    def capture(self):
        log = (self.directory / "video-ffmpeg.log").open("w")
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(self.directory / "external-camera.mp4"),
            ],
            stdin=subprocess.PIPE,
            stderr=log,
        )
        try:
            with (self.directory / "video-frames.jsonl").open("w") as timestamps:
                while not self.closed.is_set():
                    start = time.monotonic()
                    jpeg = self.console.request("/api/camera/external")
                    process.stdin.write(jpeg)
                    timestamps.write(
                        json.dumps({"frame": self.frames, "timestamp": time.time()}) + "\n"
                    )
                    self.frames += 1
                    self.closed.wait(max(0, 1 / self.fps - (time.monotonic() - start)))
        except Exception as error:
            self.error = str(error)
        finally:
            process.stdin.close()
            if process.wait(timeout=10):
                self.error = "ffmpeg failed; see video-ffmpeg.log"
            log.close()

    def close(self):
        self.closed.set()
        self.thread.join(timeout=16)
        return {
            "frames": self.frames,
            "fps": self.fps,
            "error": self.error,
            "file": "external-camera.mp4",
        }
