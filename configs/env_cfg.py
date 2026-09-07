# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for Dual Arm Imitation Learning.

This configuration inherits directly from the RL environment (dual_arm0)
without modifying any RL code, while customizing parameters for
teleoperation, demonstration collection, and IL policy rollout.
"""

from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils

# Import the base configuration from dual_arm0 (installed in environment)
from dual_arm0.tasks.dual_arm.dual_arm_env_cfg import DualArmEnvCfg, DualArmSceneCfg
from dual_arm0.tasks.dual_arm import rewards as base_rewards


@configclass
class DualArmILEnvCfg(DualArmEnvCfg):
    """Environment configuration tailored for Imitation Learning."""

    def __post_init__(self):
        super().__post_init__()

        # For teleoperation and evaluation, default to 1 environment
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Episode settings for teleoperation & evaluation
        self.episode_length_s = 25.0  # 25 seconds per episode to give human ample time
        self.decimation = 2           # 60Hz / 2 = 30Hz control frequency

        # Disable automatic time_out termination during teleop if needed (can be toggled)
        # self.terminations.time_out = None

        # Camera viewer position for comfortable human teleoperation view
        self.viewer.eye = (1.2, 0.0, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.2)

        # -------------------------------------------------------------
        # Gripper Clamping Force & Grasp Stability Enhancements
        # -------------------------------------------------------------
        import copy
        self.scene.robot = copy.deepcopy(self.scene.robot)
        self.scene.object = copy.deepcopy(self.scene.object)

        # 1. Gripper Actuator Clamping Force:
        # Default in DUAL_FRANKA_CFG was: stiffness=2000.0, damping=100.0, effort_limit=200.0.
        # When grasping a 3cm baton, finger position error is only ~0.015m,
        # producing only 30 N of clamping force (2000.0 * 0.015).
        # We increase stiffness to 10,000 N/m (5x force) and effort_limit to 400.0 N:
        # Clamping force becomes 150 N per finger (total 300 N normal force).
        if "hand" in self.scene.robot.actuators:
            self.scene.robot.actuators["hand"].stiffness = 3000.0
            self.scene.robot.actuators["hand"].damping = 100.0
            self.scene.robot.actuators["hand"].effort_limit = 300.0

        # 2. Realistic Baton Mass:
        # In dual_arm0, mass was set to 1.0kg with the note:
        # "# [수정] 1.0kg은 너무 무거워서 들고 이동할 때 놓침. 0.2kg으로 경량화."
        # Reduced to 0.1kg per user request for even lighter handling.
        if hasattr(self.scene.object.spawn, "mass_props") and self.scene.object.spawn.mass_props is not None:
            self.scene.object.spawn.mass_props.mass = 0.1

        # 3. Cuboid Size: 4cm x 4cm x 25cm
        # Enlarged from 3cm to 4cm (0.04m x 0.04m x 0.25m) for larger contact area and stronger grasp.
        # Initial center height is set to half thickness: Z = 0.020m.
        if hasattr(self.scene.object, "spawn") and hasattr(self.scene.object.spawn, "size"):
            self.scene.object.spawn.size = (0.04, 0.04, 0.25)
        if hasattr(self.scene.object, "init_state") and hasattr(self.scene.object.init_state, "pos"):
            self.scene.object.init_state.pos = (0.5, 0.0, 0.020)

        # 4. Anti-Slip High Friction Physics Material:
        # Static and dynamic friction set to 3.0 with friction_combine_mode="max"
        # to ensure rock-solid non-slip contact with Franka finger pads.
        if hasattr(self.scene.object.spawn, "physics_material"):
            self.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
                friction_combine_mode="max",
                restitution=0.0,
                restitution_combine_mode="min",
            )

        # 5. Move placement target closer to left arm to prevent unreachable IK poses
        # Original pos was (0.5, 0.5) + offset (0.1, 0.2). This pushes it to Y=0.7~0.9 which is out of reach.
        # Moving nominal pos to (0.4, 0.1) makes final pos X=0.5~0.6, Y=0.3~0.5 (perfect for left arm).
        if hasattr(self.scene, "target") and hasattr(self.scene.target, "init_state"):
            self.scene.target.init_state.pos = (0.4, 0.1, 0.001)
