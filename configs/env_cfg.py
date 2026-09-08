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

from isaaclab.sensors import CameraCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm


@configclass
class VisuomotorImageObsCfg(ObsGroup):
    """Image observations for the Visuomotor policy."""
    rgb = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb"},
    )
    
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

import dual_arm0.tasks.dual_arm.observations as custom_obs
@configclass
class VisuomotorProprioCfg(ObsGroup):
    """Proprioceptive observations for policy network."""
    # Joint positions and velocities
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
    
    # TCP poses (no object-relative terms)
    pick_tcp_pos = ObsTerm(func=custom_obs.pick_tcp_pos_w, params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"})
    pick_tcp_quat = ObsTerm(func=custom_obs.pick_tcp_quat_w, params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"})
    place_tcp_pos = ObsTerm(func=custom_obs.place_tcp_pos_w, params={"asset_name": "robot", "place_hand_regex": "panda_hand$"})
    place_tcp_quat = ObsTerm(func=custom_obs.place_tcp_quat_w, params={"asset_name": "robot", "place_hand_regex": "panda_hand$"})
    
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class VisuomotorObsCfg:
    image: VisuomotorImageObsCfg = VisuomotorImageObsCfg()
    policy: VisuomotorProprioCfg = VisuomotorProprioCfg()


@configclass
class VisuomotorSceneCfg(DualArmSceneCfg):
    # Explicitly define the camera as a dataclass field so InteractiveScene spawns it!
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955
        ),
        offset=CameraCfg.OffsetCfg(pos=(1.2, 0.0, 1.0), rot=(0.5, 0.5, 0.5, 0.5), convention="ros"),
    )

@configclass
class DualArmILEnvCfg(DualArmEnvCfg):
    """Environment configuration tailored for Imitation Learning."""
    
    # Define at the class level so dataclasses.fields() detects it!
    observations: VisuomotorObsCfg = VisuomotorObsCfg()
    scene: VisuomotorSceneCfg = VisuomotorSceneCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Override the configs AFTER the parent class initializes
        self.observations = VisuomotorObsCfg()
        
        # Re-initialize the scene using our new class that has the camera
        self.scene = VisuomotorSceneCfg()

        # For teleoperation and evaluation, default to 1 environment
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Episode settings for teleoperation & evaluation
        self.episode_length_s = 25.0  # 25 seconds per episode to give human ample time
        self.decimation = 2           # 60Hz / 2 = 30Hz control frequency

        # Camera viewer position for comfortable human teleoperation view
        self.viewer.eye = (1.2, 0.0, 1.0)
        self.viewer.lookat = (0.35, 0.0, 0.2)

        # -------------------------------------------------------------
        # Gripper Clamping Force & Grasp Stability Enhancements
        # -------------------------------------------------------------
        import copy
        self.scene.robot = copy.deepcopy(self.scene.robot)
        self.scene.object = copy.deepcopy(self.scene.object)

        if "hand" in self.scene.robot.actuators:
            self.scene.robot.actuators["hand"].stiffness = 3000.0
            self.scene.robot.actuators["hand"].damping = 100.0
            self.scene.robot.actuators["hand"].effort_limit = 300.0

        if hasattr(self.scene.object.spawn, "mass_props") and self.scene.object.spawn.mass_props is not None:
            self.scene.object.spawn.mass_props.mass = 0.1

        if hasattr(self.scene.object, "spawn") and hasattr(self.scene.object.spawn, "size"):
            self.scene.object.spawn.size = (0.04, 0.04, 0.25)
        if hasattr(self.scene.object, "init_state") and hasattr(self.scene.object.init_state, "pos"):
            self.scene.object.init_state.pos = (0.5, 0.0, 0.020)

        if hasattr(self.scene.object.spawn, "physics_material"):
            self.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=3.0,
                friction_combine_mode="max",
                restitution=0.0,
                restitution_combine_mode="min",
            )

        if hasattr(self.scene, "target") and hasattr(self.scene.target, "init_state"):
            self.scene.target.init_state.pos = (0.5, 0.5, 0.001)
            if hasattr(self.scene.target, "spawn") and hasattr(self.scene.target.spawn, "radius"):
                self.scene.target.spawn.radius = 0.125

        if hasattr(self, "events") and hasattr(self.events, "reset_target"):
            self.events.reset_target.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0)}


