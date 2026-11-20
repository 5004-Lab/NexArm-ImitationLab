"""Read NexArm servo configuration in normal firmware mode, with the console stopped.

Only READ commands are sent. This script never changes torque, mode, limits, or goals.
Protocol: Hiwonder/NexArm ros_robot_controller_sdk.py.
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, "/home/ztl/nexarm-console")
    import serial
    from nexarm.linux_serial import initialize_generic_ch340
    from nexarm.server import Frames, packet

    report = {"timestamp": time.time(), "writes_to_registers": False, "servos": {}}
    link = serial.Serial(baudrate=1_000_000, timeout=0.02, exclusive=True)
    link.port = args.port
    link.dtr = link.rts = False
    initialize_generic_ch340(args.port)
    link.open()
    try:
        time.sleep(6)  # The ESP32 boot interval also used by the running console.
        frames = Frames()

        def query(device, command, payload=b"", timeout=0.4):
            link.reset_input_buffer()
            link.write(packet(command, payload, device))
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                for frame in frames.feed(link.read(link.in_waiting or 1)):
                    if (device == 255 and frame[2] in (255, 0x5A) and frame[4] == command) or (
                        device != 255 and frame[2] == device
                    ):
                        return {
                            "device": frame[2],
                            "status_or_command": frame[4],
                            "payload": list(frame[5:-1]),
                        }
            return None

        report["firmware_reply"] = query(255, 1)
        report["voltage_reply"] = query(255, 2)
        if report["voltage_reply"] is None:
            raise RuntimeError(
                "Normal firmware mode unavailable; reset both controllers before reading servo configuration"
            )
        for servo_id in range(1, 7):
            values = {}
            for label, register, length, fmt in (
                ("mode", 33, 1, "<B"),
                ("torque_enabled", 40, 1, "<B"),
                ("position", 56, 2, "<h"),
                ("voltage_raw", 62, 1, "<B"),
                ("temperature_c", 63, 1, "<B"),
            ):
                reply = query(servo_id, 2, bytes([register, length]))
                values[label] = reply
                if reply and len(reply["payload"]) == length:
                    reply["value"] = struct.unpack(fmt, bytes(reply["payload"]))[0]
            for label, command in (("overload", 71), ("max_torque", 75), ("angle_limits", 77)):
                values[label] = query(255, command, bytes([servo_id]))
            report["servos"][str(servo_id)] = values
            print(json.dumps({"servo_id": servo_id, **values}), flush=True)
    finally:
        link.close()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
