# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Dual Arm Teleoperation Controller for Isaac Sim.

Provides SE(3) task-space control for both Franka arms with an Active-Arm Toggle (Tab key),
Differential IK calculation, and episode recording controls (Confirm/Discard/Reset).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import carb
import omni


class DualArmTeleopController:
    """Keyboard controller for Dual Franka arms in Isaac Lab.

    Key bindings:
        -------------------------------------------------------------
        [Arm Selection]
        TAB              : Toggle active arm (Right Arm <-> Left Arm)
        1                : Force select Right Arm (Pick Arm)
        2                : Force select Left Arm (Place Arm)

        [Task-Space Movement (Active Arm)]
        W / S            : Move Forward / Backward (+X / -X)
        A / D            : Move Left / Right (+Y / -Y)
        Q / E            : Move Up / Down (+Z / -Z)
        Z / X            : Roll (+ / -)
        T / G            : Pitch (+ / -)
        C / V            : Yaw (+ / -)

        [Gripper Control]
        SPACE or K       : Toggle Gripper of active arm (Open <-> Close)
        O                : Open active gripper
        P                : Close active gripper

        [Data Collection Workflow]
        ENTER or Y       : Confirm & Save current episode to HDF5
        BACKSPACE or N   : Discard current episode and reset
        R                : Reset environment

        [Sensitivity]
        UP / DOWN        : Increase / Decrease translation sensitivity
        -------------------------------------------------------------
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device

        # Sensitivities
        self.pos_step = 0.015   # 1.5 cm per press / tick
        self.rot_step = 0.03    # ~1.7 degrees per press / tick

        # Active arm tracking: "right" or "left"
        self.active_arm = "right"

        # Gripper states: True = closed (-1.0), False = open (+1.0)
        self.right_gripper_closed = False
        self.left_gripper_closed = False

        # Delta accumulators
        self.delta_pos = np.zeros(3, dtype=np.float32)
        self.delta_rot = np.zeros(3, dtype=np.float32)

        # Status flags for demonstration workflow
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False

        # Acquire Omniverse input interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events with weakref
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Key state tracking for smooth continuous holding
        self._keys_down = set()

        print("[DualArmTeleopController] Initialized successfully.")
        self.print_instructions()

    def __del__(self):
        if hasattr(self, "_sub") and self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def print_instructions(self):
        print("=" * 65)
        print(" Dual Arm Teleoperation - Key Bindings")
        print("=" * 65)
        print("  TAB          : Toggle Active Arm (Current: " + self.active_arm.upper() + ")")
        print("  W / S        : Move X (+ / -)")
        print("  A / D        : Move Y (+ / -)")
        print("  Q / E        : Move Z (+ / -)")
        print("  Z / X        : Roll  (+ / -)")
        print("  T / G        : Pitch (+ / -)")
        print("  C / V        : Yaw   (+ / -)")
        print("  SPACE / K    : Toggle Active Gripper (Open <-> Close)")
        print("  ENTER / Y    : [SAVE] Confirm & Save Demo")
        print("  BACKSPACE / N: [DISCARD] Discard Demo & Reset")
        print("  R            : [RESET] Reset Environment")
        print("=" * 65)

    def reset_episode_flags(self):
        """Reset workflow flags at the beginning of each episode."""
        self.flag_save_episode = False
        self.flag_discard_episode = False
        self.flag_reset_env = False
        self.delta_pos[:] = 0.0
        self.delta_rot[:] = 0.0

    def reset(self):
        """Full reset of controller state."""
        self.reset_episode_flags()
        self.right_gripper_closed = False
        self.left_gripper_closed = False
        self.active_arm = "right"

    def _on_keyboard_event(self, event, *args, **kwargs):
        name = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_down.add(name)

            # --- Arm Switching ---
            if name == "TAB":
                self.active_arm = "left" if self.active_arm == "right" else "right"
                print(f"[Teleop] Switched active arm to: >> {self.active_arm.upper()} <<")
            elif name in ("NUM_1", "KEY_1"):
                self.active_arm = "right"
                print("[Teleop] Active arm: >> RIGHT (Pick) <<")
            elif name in ("NUM_2", "KEY_2"):
                self.active_arm = "left"
                print("[Teleop] Active arm: >> LEFT (Place) <<")

            # --- Gripper Toggles ---
            elif name in ("SPACE", "K"):
                if self.active_arm == "right":
                    self.right_gripper_closed = not self.right_gripper_closed
                    status = "CLOSED" if self.right_gripper_closed else "OPEN"
                    print(f"[Teleop] Right Gripper: {status}")
                else:
                    self.left_gripper_closed = not self.left_gripper_closed
                    status = "CLOSED" if self.left_gripper_closed else "OPEN"
                    print(f"[Teleop] Left Gripper: {status}")
            elif name == "O":
                if self.active_arm == "right":
                    self.right_gripper_closed = False
                else:
                    self.left_gripper_closed = False
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper OPEN")
            elif name == "P":
                if self.active_arm == "right":
                    self.right_gripper_closed = True
                else:
                    self.left_gripper_closed = True
                print(f"[Teleop] {self.active_arm.capitalize()} Gripper CLOSED")

            # --- Workflow Flags ---
            elif name in ("ENTER", "Y"):
                self.flag_save_episode = True
                print("\n>>> [DEMO CONFIRMED] Saving episode to dataset! <<<")
            elif name in ("BACKSPACE", "N"):
                self.flag_discard_episode = True
                print("\n>>> [DEMO DISCARDED] Episode dropped! Resetting... <<<")
            elif name == "R":
                self.flag_reset_env = True
                print("[Teleop] Reset requested.")

            # --- Sensitivity Adjust ---
            elif name == "UP":
                self.pos_step = min(0.05, self.pos_step + 0.005)
                print(f"[Teleop] Sensitivity increased: {self.pos_step*100:.1f} cm/step")
            elif name == "DOWN":
                self.pos_step = max(0.002, self.pos_step - 0.005)
                print(f"[Teleop] Sensitivity decreased: {self.pos_step*100:.1f} cm/step")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_down.discard(name)

    def get_delta_command(self) -> tuple[str, np.ndarray, np.ndarray, float, float]:
        """Poll currently held keys and compute delta position and rotation for the active arm.

        Returns:
            active_arm: "right" or "left"
            dpos: np.ndarray (3,) [dx, dy, dz] in meters
            drot: np.ndarray (3,) [droll, dpitch, dyaw] in radians
            right_gripper_val: float (-1.0 for close, +1.0 for open)
            left_gripper_val: float (-1.0 for close, +1.0 for open)
        """
        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)

        # X axis (Forward / Backward)
        if "W" in self._keys_down:
            dpos[0] += self.pos_step
        if "S" in self._keys_down:
            dpos[0] -= self.pos_step

        # Y axis (Left / Right)
        if "A" in self._keys_down:
            dpos[1] += self.pos_step
        if "D" in self._keys_down:
            dpos[1] -= self.pos_step

        # Z axis (Up / Down)
        if "Q" in self._keys_down:
            dpos[2] += self.pos_step
        if "E" in self._keys_down:
            dpos[2] -= self.pos_step

        # Roll
        if "Z" in self._keys_down:
            drot[0] += self.rot_step
        if "X" in self._keys_down:
            drot[0] -= self.rot_step

        # Pitch
        if "T" in self._keys_down:
            drot[1] += self.rot_step
        if "G" in self._keys_down:
            drot[1] -= self.rot_step

        # Yaw
        if "C" in self._keys_down:
            drot[2] += self.rot_step
        if "V" in self._keys_down:
            drot[2] -= self.rot_step

        right_grip = -1.0 if self.right_gripper_closed else 1.0
        left_grip = -1.0 if self.left_gripper_closed else 1.0

        return self.active_arm, dpos, drot, right_grip, left_grip
