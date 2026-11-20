"""Native tick mapping, bounded trajectories, and independent control heartbeat."""

import math
import time
import unittest
from unittest.mock import Mock

import numpy as np

from imitation_lab.native_deploy import ControlLease, bounded_target, checked_state, encoder_angles


class NativeDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.joints = [
            {"zero": 2048, "ticks": 626.8, "sign": 1, "min": 739, "max": 3357},
            {"zero": 2048, "ticks": 591.5, "sign": -1, "min": 1457, "max": 2976},
            {"zero": 1198, "ticks": 624.6, "sign": -1, "min": 1011, "max": 3159},
            {"zero": 1200, "ticks": 586.9, "sign": 1, "min": 813, "max": 2121},
            {"zero": 2048, "ticks": 651.9, "sign": 1, "min": 686, "max": 3410},
            {"zero": 1000, "ticks": 3200, "sign": 1, "min": 1000, "max": 2600},
        ]
        self.limits = {
            "step_deg": 1.9,
            "travel_deg": 6,
            "gripper_step_ticks": 20,
            "gripper_travel_ticks": 60,
        }

    def test_encoder_round_trip_uses_follower_order_and_gripper_ticks(self):
        pulses = [2010, 2139, 1236, 2168, 1884, 2780]
        angles = encoder_angles(pulses, self.joints)
        restored = [
            round(j["zero"] + math.radians(a) * j["ticks"] / j["sign"])
            for a, j in zip(angles, self.joints)
        ]
        self.assertEqual(restored, pulses)
        self.assertLess(angles[1], 0)
        self.assertLess(angles[2], 0)
        self.assertGreater(angles[-1], 30)  # It is an angle, not an open fraction.

    def test_repeated_model_targets_stay_within_total_travel(self):
        origin = [2010, 2139, 1236, 2168, 1884, 2780]
        prediction = [2324, 1759, 1757, 2324, 1393, 2297]
        actual = origin
        step_limit = np.array([j["ticks"] * math.pi / 180 * 1.9 for j in self.joints[:5]])
        for _ in range(30):
            target = bounded_target(prediction, actual, origin, self.joints, self.limits)
            self.assertTrue((np.abs(np.array(target[:5]) - actual[:5]) <= step_limit).all())
            self.assertLessEqual(abs(target[-1] - actual[-1]), 20)
            self.assertEqual(target[3], origin[3])  # No further movement outside wrist bound.
            self.assertLessEqual(abs(target[-1] - origin[-1]), 60)
            actual = target
        angles = np.abs(
            np.array(encoder_angles(actual, self.joints))[:5]
            - np.array(encoder_angles(origin, self.joints))[:5]
        )
        self.assertTrue((angles <= 6).all())

    def test_outside_soft_limit_returns_gradually_without_jump(self):
        origin = [2010, 2139, 1236, 2168, 1884, 2780]
        target = bounded_target(
            [2010, 2139, 1236, 2000, 1884, 2000], origin, origin, self.joints, self.limits
        )
        self.assertGreater(target[3], 2121)
        self.assertLess(target[3], 2168)
        self.assertGreater(target[-1], 2600)

    def test_heartbeat_runs_while_main_thread_is_blocked(self):
        console = Mock()
        lease = ControlLease(console, interval=0.01)
        lease.start()
        time.sleep(0.075)  # Simulates blocked inference; the lease thread must progress.
        lease.check()
        lease.close()
        self.assertGreaterEqual(lease.count, 4)
        self.assertEqual(console.command.call_args.args, ("stop",))

    def test_controller_stop_reason_is_propagated(self):
        console = Mock()
        console.state.return_value = {
            "connected": True,
            "armed": False,
            "result": "J3 tracking error",
        }
        with self.assertRaisesRegex(RuntimeError, "J3 tracking error"):
            checked_state(console, Mock())

    def test_heartbeat_failure_is_not_hidden(self):
        console = Mock()
        console.command.side_effect = RuntimeError("ownership lost")
        lease = ControlLease(console, interval=0.01)
        lease.start()
        lease.thread.join(1)
        with self.assertRaisesRegex(RuntimeError, "ownership lost"):
            lease.check()
