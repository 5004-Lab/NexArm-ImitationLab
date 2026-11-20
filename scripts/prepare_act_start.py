"""Explicit single-axis setup movement; this is not an ACT prediction."""

import argparse
import json
import sys
from pathlib import Path

from imitation_lab.native_deploy import ControlLease, encoder_angles, wait_motion
from imitation_lab.robot import Console
from imitation_lab.trial_video import TrialVideo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shoulder-delta-ticks", type=int, required=True)
    parser.add_argument("--console-root", default="/home/ztl/nexarm-console")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, args.console_root)
    from nexarm import kinematics

    console = Console("http://127.0.0.1:8765", kinematics.JOINTS)
    state = console.state()
    if not state["connected"] or not state["power_ok"]:
        raise ValueError("Restore console feedback before preparation")
    start = [v[0] for v in state["feedback"]]
    target = start.copy()
    target[1] += args.shoulder_delta_ticks
    kinematics.check_path(start, target, kinematics.joint_limits(start))
    report = {
        "purpose": "manual single-axis preparation, not ACT output",
        "start": start,
        "target": target,
        "tcp_before": kinematics.tcp(start),
        "tcp_target": kinematics.tcp(target),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    if args.execute:
        lease = ControlLease(console)
        video = TrialVideo(console, directory)
        lease.start()
        video.start()
        try:
            console.command(
                "move-joints",
                {
                    "angles": encoder_angles(target, kinematics.JOINTS),
                    "completion_tolerance_ticks": 10,
                },
            )
            state = wait_motion(console, lease, "STOP", timeout=12)
            report["feedback_after"] = [v[0] for v in state["feedback"]]
            report["controller_result"] = state["result"]
        finally:
            try:
                lease.close()
            finally:
                report["video"] = video.close()
                (directory / "summary.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)
                )
    else:
        (directory / "plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
