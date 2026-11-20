"""Start and observe robot-side I/O from the GPU server over SSH."""

import argparse
import shlex
import subprocess
import time
from pathlib import Path

ROBOT_ROOT = "/home/ztl/NexArm-ImitationLab"
KEY = "/home/wangji/.ssh/nexarm_robot_ed25519"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("status", "watch", "shadow", "execute", "stop", "clear-stop"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument(
        "--samples", type=int, default=0, help="Watch samples; zero means until Ctrl+C"
    )
    parser.add_argument("--output", default=None, help="Robot-side output directory")
    parser.add_argument("--host", default="ztl@10.206.95.245")
    parser.add_argument("--key", default=KEY)
    args = parser.parse_args()
    ssh = [
        "ssh",
        "-i",
        str(Path(args.key).expanduser()),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        args.host,
    ]
    if args.stop or args.clear_stop:
        action = "unlink(missing_ok=True)" if args.clear_stop else "touch()"
        code = f"from pathlib import Path; Path({(ROBOT_ROOT + '/STOP')!r}).{action}"
        command = shlex.join(["python3", "-c", code])
    elif args.status or args.watch:
        samples = args.samples if args.watch else 1
        code = f"""import json,time,urllib.request
names=['J1 base','J2 shoulder','J3 ELBOW','J4 wrist pitch','J5 wrist roll','J6 gripper']
i=0
while {samples} == 0 or i < {samples}:
 with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=3) as r:s=json.load(r)
 print(time.strftime('%H:%M:%S'),'connected=',s['connected'],'moving=',s['moving'],flush=True)
 if s['feedback']:
  for n,p,a in zip(names,s['feedback'],s['angles']):print('%-15s ticks=%4d  angle=%7.2f deg'%(n,p[0],a),flush=True)
 print('controller:',s['result'],flush=True)
 i+=1
 if {samples} == 0 or i < {samples}:time.sleep(1)
"""
        command = shlex.join(["python3", "-u", "-c", code])
    else:
        output = args.output or "outputs/gpu-act-" + time.strftime("%Y%m%d-%H%M%S")
        command = (
            "cd "
            + shlex.quote(ROBOT_ROOT)
            + " && "
            + shlex.join(
                [
                    ".venv/bin/python",
                    "-u",
                    "-m",
                    "imitation_lab",
                    "deploy-act",
                    "--cycles",
                    str(args.cycles),
                    "--output",
                    output,
                ]
                + (["--execute"] if args.execute else [])
            )
        )
        print("GPU ACT inference: http://10.206.58.217:8790", flush=True)
        print("Robot output:", ROBOT_ROOT + "/" + output, flush=True)
    try:
        result = subprocess.run(ssh + [command])
    except KeyboardInterrupt:
        if args.execute:
            subprocess.run(ssh + ["touch " + ROBOT_ROOT + "/STOP"], timeout=5)
        raise SystemExit(130)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
