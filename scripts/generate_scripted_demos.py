# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted Automatic Demonstration Generator for Dual Arm Handover Task.

Generates high-quality, optimal demonstration trajectories automatically using
Waypoint State Machine and Differential Inverse Kinematics.

Features:
- 100% automated: No human teleoperation or manual effort required.
- Robust Closed-Loop IK: DLS position control with joint limit protection.
- Exact Action Inversion: Automatically accounts for Isaac Lab's default joint offsets and action scales.
- Direct export to HDF5: Ready for immediate training with BC, Diffusion Policy, or ACT.

Usage:
    # Fast Headless generation (recommended):
    python scripts/generate_scripted_demos.py --num_demos=500 --headless

    # GUI mode (visualize robot motion):
    python scripts/generate_scripted_demos.py --num_demos=10
"""

import argparse
import os
import sys
import time
import h5py
import numpy as np
import torch

# Add project root and parent to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate scripted demonstrations for Dual Arm Handover.")
parser.add_argument("--num_demos", type=int, default=50, help="Number of successful demonstrations to generate.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to output HDF5 dataset.",
)
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps before episode timeout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg


def solve_pose_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
    q_current: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    k_null: float = 0.5,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics with 7-DOF nullspace elbow flare projection.

    Args:
        jacobian: Full Jacobian of shape (1, 6, 7).
        delta_pose: Desired 6D delta [dx, dy, dz, wx, wy, wz] of shape (1, 6).
        damping: Damping factor lambda.
        q_current: Current joint positions of shape (1, 7).
        q_nominal: Desired nominal joint posture for nullspace projection (1, 7).
        k_null: Nullspace projection gain.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)
    j_pinv = torch.matmul(j.transpose(0, 1), inv_term)  # (7, 6)
    delta_q = torch.matmul(j_pinv, e)  # (7,)

    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        null_proj = eye_n - torch.matmul(j_pinv, j)  # (7, 7)
        q_err = (q_nominal - q_current).squeeze(0)  # (7,)
        delta_q_null = torch.matmul(null_proj, k_null * q_err)  # (7,)
        delta_q = delta_q + delta_q_null

    return delta_q.unsqueeze(0)



def get_tcp_jacobian(jacobian: torch.Tensor, wrist_pos: torch.Tensor, tcp_pos: torch.Tensor) -> torch.Tensor:
    r = tcp_pos - wrist_pos  # (N, 3)
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zero = torch.zeros_like(rx)
    r_skew = torch.stack([
        torch.stack([zero, -rz, ry], dim=-1),
        torch.stack([rz, zero, -rx], dim=-1),
        torch.stack([-ry, rx, zero], dim=-1)
    ], dim=1)  # (N, 3, 3)
    j_v = jacobian[:, :3, :]   # (N, 3, 7)
    j_w = jacobian[:, 3:6, :]  # (N, 3, 7)
    j_v_tcp = j_v - torch.bmm(r_skew, j_w)
    return torch.cat([j_v_tcp, j_w], dim=1)  # (N, 6, 7)


def compute_desired_grasp_rot(
    wrist_quat: torch.Tensor,
    cube_z_dir: torch.Tensor,
    gain: float = 0.5,
    max_rot_step: float = 0.04,
    directed: bool = False,
) -> torch.Tensor:
    """Compute angular error vector to align Franka gripper perpendicular to baton.

    Ensures:
    1. Local Z axis (palm normal) points vertically downwards [0, 0, -1].
    2. Local X axis (palm lateral) is parallel to the baton longitudinal axis.
    3. Local Y axis (finger opening/closing) is perpendicular to the baton,
       pinching the 3cm thickness cleanly.

    Args:
        wrist_quat: Hand orientation quaternion (1, 4) in (w, x, y, z).
        cube_z_dir: Baton length direction unit vector (1, 3).
        gain: Proportional convergence gain.
        max_rot_step: Maximum angular velocity step in radians.
        directed: If True, do not use 180-degree symmetry. Align exactly to cube_z_dir.

    Returns:
        e_rot_step: (1, 3) angular delta vector [wx, wy, wz].
    """
    w = wrist_quat[:, 0]
    x = wrist_quat[:, 1]
    y = wrist_quat[:, 2]
    z = wrist_quat[:, 3]

    # Current rotation matrix columns in world coordinates
    x_curr = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ], dim=-1)

    y_curr = torch.stack([
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], dim=-1)

    z_curr = torch.stack([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dim=-1)

    # Desired Z axis: straight down
    z_des = torch.tensor([[0.0, 0.0, -1.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    # Baton horizontal projection
    baton_xy = cube_z_dir.clone()
    baton_xy[:, 2] = 0.0
    norm_xy = torch.norm(baton_xy, dim=-1, keepdim=True)
    if norm_xy.item() > 1e-4:
        x_cand = baton_xy / norm_xy
    else:
        x_cand = torch.tensor([[1.0, 0.0, 0.0]], device=wrist_quat.device, dtype=wrist_quat.dtype)

    if directed:
        x_des = x_cand
    else:
        # Align with whichever 180-degree symmetric direction is closer to current x_curr
        dot = torch.sum(x_curr * x_cand, dim=-1, keepdim=True)
        sign = torch.sign(dot + 1e-6)
        x_des = sign * x_cand

    # Desired Y axis = Z_des x X_des (finger pinching axis perpendicular to baton)
    y_des = torch.cross(z_des, x_des, dim=-1)

    # Orientation error via Lie algebra so(3) cross-product error
    e_rot = 0.5 * (
        torch.cross(x_curr, x_des, dim=-1) +
        torch.cross(y_curr, y_des, dim=-1) +
        torch.cross(z_curr, z_des, dim=-1)
    )

    # Apply gain and clamp
    e_rot_step = e_rot * gain
    err_norm = torch.norm(e_rot_step)
    if err_norm > max_rot_step:
        e_rot_step = e_rot_step * (max_rot_step / err_norm)

    return e_rot_step


def compute_tcp(wrist_pos: torch.Tensor, wrist_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Franka TCP position and Z-axis direction from wrist pose."""
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    return tcp_pos, z_dir


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list):
    os.makedirs(os.path.dirname(os.path.abspath(hdf5_path)), exist_ok=True)
    mode = "a" if os.path.exists(hdf5_path) else "w"
    with h5py.File(hdf5_path, mode) as f:
        data_group = f.require_group("data")
        demo_group = data_group.create_group(f"demo_{ep_idx}")

        obs_array = np.array(observations, dtype=np.float32)
        act_array = np.array(actions, dtype=np.float32)
        rew_array = np.array(rewards, dtype=np.float32)

        demo_group.create_dataset("obs", data=obs_array, compression="gzip")
        demo_group.create_dataset("actions", data=act_array, compression="gzip")
        demo_group.create_dataset("rewards", data=rew_array, compression="gzip")
        demo_group.attrs["num_samples"] = len(act_array)

        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


# State Machine Phase Constants
PHASE_NAMES = [
    "INIT",
    "RIGHT_HOVER",
    "RIGHT_DESCEND",
    "RIGHT_GRASP",
    "RIGHT_LIFT",
    "RIGHT_HANDOVER",
    "LEFT_APPROACH",
    "LEFT_GRASP",
    "RIGHT_RELEASE",
    "RIGHT_LIFT_CLEAR",
    "RIGHT_RETREAT",
    "LEFT_HOVER_TARGET",
    "LEFT_LOWER_TARGET",
    "LEFT_RELEASE",
    "LEFT_RETREAT",
    "SUCCESS",
]
PHASE_INIT = 0
PHASE_RIGHT_HOVER = 1
PHASE_RIGHT_DESCEND = 2
PHASE_RIGHT_GRASP = 3
PHASE_RIGHT_LIFT = 4
PHASE_RIGHT_HANDOVER = 5
PHASE_LEFT_APPROACH = 6
PHASE_LEFT_GRASP = 7
PHASE_RIGHT_RELEASE = 8
PHASE_RIGHT_LIFT_CLEAR = 9
PHASE_RIGHT_RETREAT = 10
PHASE_LEFT_HOVER_TARGET = 11
PHASE_LEFT_LOWER_TARGET = 12
PHASE_LEFT_RELEASE = 13
PHASE_LEFT_RETREAT = 14
PHASE_SUCCESS = 15


def main():
    cfg = DualArmILEnvCfg()
    cfg.sim.device = args_cli.device

    print("[Scripted Demo] Initializing Isaac Lab Environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target"]

    # Body Indices
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Joint Indices in Articulation
    left_arm_joint_ids, left_arm_names = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, right_arm_names = robot.find_joints("panda_joint[1-7]_0")

    # Gripper finger joint indices (for real-time grip width and fake grasp verification)
    left_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]$")
    right_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]_0")

    # Safe standby postures (arm base and elbow rotated away from center workspace)
    left_standby_joints = robot.data.default_joint_pos[:, left_arm_joint_ids].clone()
    left_standby_joints[:, 0] = 0.60  # +34 deg, outward to +Y quadrant

    right_standby_joints = robot.data.default_joint_pos[:, right_arm_joint_ids].clone()
    right_standby_joints[:, 0] = -0.60  # -34 deg, outward to -Y quadrant

    right_standby_joints[:, 4] += 0.5
    left_standby_joints[:, 4] -= 0.5

    # Nominal joint postures for 7-DOF nullspace elbow flaring
    q_nominal_left = left_standby_joints.clone()
    q_nominal_right = right_standby_joints.clone()

    # Fixed world orientations for stable, jitter-free transport
    fixed_z_down = torch.tensor([[0.0, 0.0, -1.0]], device=args_cli.device)
    fixed_x_along = torch.tensor([[1.0, 0.0, 0.0]], device=args_cli.device)

    # Shift joint IDs for Jacobian lookup (0 for fixed base robot)
    num_base_dofs = getattr(robot, "num_base_dofs", 0)
    jacobi_right_joint_ids = [j + num_base_dofs for j in right_arm_joint_ids]
    jacobi_left_joint_ids = [j + num_base_dofs for j in left_arm_joint_ids]

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]
    print(f"[Scripted Demo] Action Manager 'arm_action' controls {len(arm_term._joint_ids)} joints.")
    print(f"  Scale : {arm_term._scale}")

    # Joint limits for safety clamping
    soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]

    # Check existing dataset
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

    print(f"[Scripted Demo] Existing demonstrations: {existing_demos}")
    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    start_time = time.time()
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()

        # Initialize commanded joint targets from the robot's safe standby positions
        commanded_left = left_standby_joints.clone()
        commanded_right = right_standby_joints.clone()

        left_gripper_cmd = 1.0   # Open (+1.0)
        right_gripper_cmd = 1.0  # Open (+1.0)

        phase = PHASE_INIT
        phase_timer = 0
        latched_align_sign = None
        latched_cube_z = None
        target_cube_z = None
        target_cube_z_for_wrist = None
        latched_left_grasp_offset = None

        ep_obs = []
        ep_actions = []
        ep_rewards = []

        print(f"\n>>> Generating Scripted Demo #{collected_count} (Target: {target_count}) <<<")

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            phase_timer += 1

            # Object and Target Poses
            obj_pos = obj.data.root_pos_w.clone()        # (1, 3)
            obj_quat = obj.data.root_quat_w.clone()      # (1, 4)
            target_pos = target.data.root_pos_w.clone()  # (1, 3)

            # Fail-fast: If baton drops during mid-air phases (Handover to Hover Target), abort episode
            if 5 <= phase <= 11 and obj_pos[0, 2].item() < 0.05:
                print(f"  [X] Dropped baton in mid-air at phase {PHASE_NAMES[phase]}! Discarding episode...")
                break

            # Franka Wrist and TCP Poses
            wrist_pos_r = robot.data.body_pos_w[:, right_hand_idx]
            wrist_quat_r = robot.data.body_quat_w[:, right_hand_idx]
            tcp_pos_r, z_dir_r = compute_tcp(wrist_pos_r, wrist_quat_r)

            wrist_pos_l = robot.data.body_pos_w[:, left_hand_idx]
            wrist_quat_l = robot.data.body_quat_w[:, left_hand_idx]
            tcp_pos_l, z_dir_l = compute_tcp(wrist_pos_l, wrist_quat_l)

            # Real-time gripper widths (for fake grasp detection)
            right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1).item()
            left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1).item()

            # Baton Geometry: Long axis vector (Cuboid Z-axis in world frame)
            ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
            cube_z = torch.stack([
                2.0 * (ox * oz + ow * oy),
                2.0 * (oy * oz - ow * ox),
                1.0 - 2.0 * (ox * ox + oy * oy),
            ], dim=-1)

            # Latch grasp alignment sign at episode start to prevent sign-flipping glitch
            if latched_align_sign is None:
                latched_align_sign = torch.sign(cube_z[:, 1:2] + 1e-6).clone()
                latched_cube_z = cube_z.clone()
                target_cube_z = torch.tensor([[0.0, latched_align_sign.item(), 0.0]], device=args_cli.device)

            # Optimal Grasp Targets:
            # Distance 0.055m (5.5cm from center) reduces cantilever droop torque by 35%
            # while leaving 11cm of clearance between right and left grippers!
            pick_target = obj_pos - latched_align_sign * 0.055 * latched_cube_z
            if latched_align_sign is not None:
                place_target = obj_pos + latched_align_sign * 0.055 * target_cube_z
            else:
                place_target = obj_pos + latched_align_sign * 0.055 * latched_cube_z

            # Handover zone (between both arms)
            handover_pos = env.scene.env_origins + torch.tensor([[0.30, 0.0, 0.20]], device=args_cli.device)
            # Left arm pre-handover hover position: 20cm in +Y, 12cm above handover zone
            left_wait_pos = handover_pos.clone()
            left_wait_pos[:, 1] += 0.20
            left_wait_pos[:, 2] += 0.12

            delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
            delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
            delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
            
            k_null_r = 0.05
            k_null_l = 0.05
            damping_r = 0.05
            damping_l = 0.05

            # --- State Machine Logic ---
            if phase == PHASE_INIT:
                right_gripper_cmd = 1.0
                left_gripper_cmd = 1.0
                # Smoothly swing left arm to safe standby posture (+Y quadrant)
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_HOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HOVER:
                # Left arm holds safe standby posture away from center
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                hover_tgt = pick_target.clone()
                hover_tgt[:, 2] = pick_target[:, 2] + 0.08
                err = hover_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(hover_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(hover_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, directed=True)
                if (dist_xy < 0.025 and dist_z < 0.03) or phase_timer > 90:
                    phase = PHASE_RIGHT_DESCEND
                    phase_timer = 0

            elif phase == PHASE_RIGHT_DESCEND:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Descend to center of cuboid (TCP Z=0.025, to avoid fingertips crashing into table)
                descend_tgt = pick_target.clone()
                descend_tgt[:, 2] = torch.clamp(pick_target[:, 2], min=0.025)
                err = descend_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(descend_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(descend_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                reached = (dist_xy < 0.010 and dist_z < 0.005)
                if reached or phase_timer > 160:
                    if reached:
                        print(f"  [Right Grasp Reached!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    else:
                        print(f"  [Right Grasp Timeout!] Step {step:03d} | R_TCP_Z: {tcp_pos_r[0,2]:.3f} | Target_Z: {descend_tgt[0,2]:.3f} (diff: {dist_z.item()*1000:.1f}mm)")
                    phase = PHASE_RIGHT_GRASP
                    phase_timer = 0

            elif phase == PHASE_RIGHT_GRASP:
                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                # Close right gripper tightly around baton
                right_gripper_cmd = -1.0
                if phase_timer > 25:
                    if right_gripper_width > 0.025:
                        print(f"  [Right Grasp Verified!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm")
                        
                        w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                        x_curr = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                        latched_wrist_sign = latched_align_sign.clone()
                        target_cube_z_for_wrist = latched_wrist_sign * target_cube_z
                        
                        phase = PHASE_RIGHT_LIFT
                        phase_timer = 0
                    else:
                        print(f"  [Right Grasp Missed!] Step {step:03d} | Width: {right_gripper_width*1000:.1f}mm < 25mm. Retrying descend...")
                        right_gripper_cmd = 1.0
                        if phase_timer > 40:
                            phase = PHASE_RIGHT_DESCEND
                            phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT:

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                # Lift to stable handover height Z = 0.18m with locked horizontal orientation
                lift_tgt = pick_target.clone()
                lift_tgt[:, 2] = 0.18
                err = lift_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.025 or phase_timer > 70:
                    phase = PHASE_RIGHT_HANDOVER
                    phase_timer = 0

            elif phase == PHASE_RIGHT_HANDOVER:
                # Right arm transports baton to handover zone and slowly twists 90 degrees
                # Completely eliminate nullspace and massively increase damping to safely pass through wrist singularity
                k_null_r = 0.0
                damping_r = 0.25
                err = handover_pos - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                # Extremely slow and smooth rotation to prevent inertial shaking of the 25cm baton
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.1, max_rot_step=0.020, directed=True)

                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)

                # Wait until BOTH position and orientation (90-degree twist) are fully reached
                # Orientation is aligned when the right wrist's local X-axis is strictly parallel to target_cube_z_for_wrist
                w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                x_curr_r = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                rot_aligned = (1.0 - torch.sum(x_curr_r * target_cube_z_for_wrist, dim=-1)) < 0.005
                
                if (dist < 0.03 and rot_aligned.item()) or phase_timer > 400:
                    phase = PHASE_LEFT_APPROACH
                    phase_timer = 0

            elif phase == PHASE_LEFT_APPROACH:
                # Right arm holds baton rock-solid at handover zone
                k_null_r = 0.05
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)

                # Left arm descends: first move horizontally to align above, then straight down
                left_grasp_tgt = place_target.clone()
                left_grasp_tgt[:, 2] = place_target[:, 2] - 0.006
                dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2])
                
                if dist_xy > 0.03:
                    curr_tgt = left_grasp_tgt.clone()
                    curr_tgt[:, 2] = left_wait_pos[:, 2]
                else:
                    curr_tgt = left_grasp_tgt.clone()
                    
                err = curr_tgt - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(left_grasp_tgt[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                reached_l = (dist_xy < 0.025 and dist_z < 0.015)
                if reached_l or phase_timer > 160:
                    if reached_l:
                        print(f"  [Left Handover Reached!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    else:
                        print(f"  [Left Handover Timeout!] Step {step:03d} | L_TCP_Z: {tcp_pos_l[0,2]:.3f} | Place_Z: {place_target[0,2]:.3f}")
                    phase = PHASE_LEFT_GRASP
                    phase_timer = 0

            elif phase == PHASE_LEFT_GRASP:
                k_null_r = 0.05
                # Right arm continues holding
                right_gripper_cmd = -1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                # Left arm closes gripper
                left_gripper_cmd = -1.0
                if phase_timer > 35:
                    if left_gripper_width > 0.025:
                        print(f"  [Left Grasp Verified!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm")
                        latched_left_grasp_offset = tcp_pos_l - obj_pos
                        phase = PHASE_RIGHT_RELEASE
                        phase_timer = 0
                    else:
                        print(f"  [Left Grasp Missed!] Step {step:03d} | Width: {left_gripper_width*1000:.1f}mm < 25mm. Retrying...")
                        left_gripper_cmd = 1.0
                        if phase_timer > 50:
                            phase = PHASE_LEFT_APPROACH
                            phase_timer = 0

            elif phase == PHASE_RIGHT_RELEASE:
                k_null_r = 0.05
                # Left arm holds tightly
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm opens gripper
                right_gripper_cmd = 1.0
                delta_r_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_r_rot = torch.zeros((1, 3), device=args_cli.device)
                if phase_timer > 15:
                    phase = PHASE_RIGHT_LIFT_CLEAR
                    phase_timer = 0

            elif phase == PHASE_RIGHT_LIFT_CLEAR:
                k_null_r = 0.05
                # Left arm firmly holds baton
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                # Right arm moves -Y (away from left arm) AND UP to avoid collision
                right_gripper_cmd = 1.0
                lift_clear_tgt = handover_pos.clone()
                lift_clear_tgt[:, 1] -= 0.15  # Move -15cm in Y (away from left arm)
                lift_clear_tgt[:, 2] = 0.35   # Lift up to Z=0.35m
                err = lift_clear_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.03 or phase_timer > 60:
                    phase = PHASE_RIGHT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_RIGHT_RETREAT:
                # Right arm retreats backwards towards base to clear the table
                retreat_tgt = lift_clear_tgt.clone()
                retreat_tgt[:, 0] -= 0.15  # Move back 15cm
                err = retreat_tgt - tcp_pos_r
                step_size = min(0.012, torch.norm(err).item())
                delta_r_pos = (err / (torch.norm(err) + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z)
                left_gripper_cmd = -1.0
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)
                right_gripper_cmd = 1.0
                # Right arm smoothly moves to safe right standby posture (-Y quadrant)
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                if phase_timer > 15:
                    phase = PHASE_LEFT_HOVER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_HOVER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Manual tuning to compensate for physical drop/slip offset (pull back from +X, +Y)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_hover = final_place_target.clone()
                tgt_hover[:, 2] = target_pos[:, 2] + 0.12
                err = tgt_hover - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.016, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                # Locked fixed horizontal orientation to eliminate rotational jitter/slipping
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])
                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])
                if (dist_xy < 0.01 and dist_z < 0.015) or phase_timer > 70:
                    phase = PHASE_LEFT_LOWER_TARGET
                    phase_timer = 0

            elif phase == PHASE_LEFT_LOWER_TARGET:
                # Right arm stays safely parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Gently lower baton onto target (Z=0.022m, soft landing 2mm above table surface)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset + manual_offset
                tgt_lower = final_place_target.clone()
                tgt_lower[:, 2] = 0.022
                err = tgt_lower - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(tgt_lower[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.008, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                if (dist < 0.015 or dist_z < 0.006) or phase_timer > 60:
                    phase = PHASE_LEFT_RELEASE
                    phase_timer = 0

            elif phase == PHASE_LEFT_RELEASE:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                # Open left gripper to release baton on target
                left_gripper_cmd = 1.0
                if phase_timer > 10:
                    phase = PHASE_LEFT_RETREAT
                    phase_timer = 0

            elif phase == PHASE_LEFT_RETREAT:
                # Right arm stays parked in right standby posture
                commanded_right = commanded_right + torch.clamp(right_standby_joints - commanded_right, min=-0.04, max=0.04)
                left_gripper_cmd = 1.0
                # Move left arm up and retreat
                tgt_up = target_pos.clone()
                tgt_up[:, 2] += 0.20
                err = tgt_up - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                if dist < 0.03 or phase_timer > 40:
                    phase = PHASE_SUCCESS
                    phase_timer = 0

            # Real-time 3D state and tracking error monitoring
            if step % 25 == 0:
                dist_pick = torch.norm(pick_target - tcp_pos_r).item() * 1000
                dist_place = torch.norm(place_target - tcp_pos_l).item() * 1000
                print(
                    f"  Step {step:03d} | {PHASE_NAMES[phase]:<18} | "
                    f"Pick:[{pick_target[0,0]:.2f},{pick_target[0,1]:.2f},{pick_target[0,2]:.2f}] | "
                    f"R_TCP:[{tcp_pos_r[0,0]:.2f},{tcp_pos_r[0,1]:.2f},{tcp_pos_r[0,2]:.2f}]({dist_pick:.1f}mm) | "
                    f"Place:[{place_target[0,0]:.2f},{place_target[0,1]:.2f},{place_target[0,2]:.2f}] | "
                    f"L_TCP:[{tcp_pos_l[0,0]:.2f},{tcp_pos_l[0,1]:.2f},{tcp_pos_l[0,2]:.2f}]({dist_place:.1f}mm)"
                )

            # --- 6D Inverse Kinematics with 7-DOF Nullspace Elbow Flare Projection ---
            jacobians = robot.root_physx_view.get_jacobians()

            delta_r_pose = torch.cat([delta_r_pos, delta_r_rot], dim=-1)
            if torch.norm(delta_r_pose) > 1e-5:
                j_r_wrist = jacobians[:, jacobi_right_hand_idx, :6, :][:, :, jacobi_right_joint_ids]
                j_r_tcp = get_tcp_jacobian(j_r_wrist, wrist_pos_r, tcp_pos_r)
                dq_r = solve_pose_dls_ik(
                    j_r_tcp,
                    delta_r_pose,
                    damping=damping_r,
                    q_current=commanded_right,
                    q_nominal=q_nominal_right,
                    k_null=k_null_r,
                )
                dq_r = torch.clamp(dq_r, min=-0.08, max=0.08)
                commanded_right += dq_r
                # Clamp within safe joint limits
                commanded_right = torch.clamp(
                    commanded_right,
                    min=soft_lower[:, right_arm_joint_ids] + 0.02,
                    max=soft_upper[:, right_arm_joint_ids] - 0.02,
                )

            delta_l_pose = torch.cat([delta_l_pos, delta_l_rot], dim=-1)
            if torch.norm(delta_l_pose) > 1e-5:
                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                
                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                if phase == PHASE_LEFT_HOVER_TARGET:
                    j_l_tcp = j_l_tcp.clone()
                    j_l_tcp[:, 3:6, :] = 0.0
                    
                dq_l = solve_pose_dls_ik(
                    j_l_tcp,
                    delta_l_pose,
                    damping=damping_l,
                    q_current=commanded_left,
                    q_nominal=q_nominal_left,
                    k_null=k_null_l,
                )
                dq_l = torch.clamp(dq_l, min=-0.08, max=0.08)
                commanded_left += dq_l
                commanded_left = torch.clamp(
                    commanded_left,
                    min=soft_lower[:, left_arm_joint_ids] + 0.02,
                    max=soft_upper[:, left_arm_joint_ids] - 0.02,
                )

            # --- Action Inversion & Mapping ---
            # 1. Update full articulation joint targets
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left
            target_all_joints[:, right_arm_joint_ids] = commanded_right

            # 2. Extract joints in the exact order arm_action expects
            arm_targets = target_all_joints[:, arm_term._joint_ids]

            # 3. Invert default offset and scale so applied == arm_targets
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            # 4. Grippers
            l_grip_t = torch.tensor([[left_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            r_grip_t = torch.tensor([[right_gripper_cmd]], dtype=torch.float32, device=args_cli.device)
            action = torch.cat([raw_arm_action, l_grip_t, r_grip_t], dim=-1)

            # Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            action_np = action.squeeze(0).detach().cpu().numpy()
            ep_obs.append(policy_obs)
            ep_actions.append(action_np)

            # Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # Check Termination / Success
            if phase == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos[0, :2] - target_pos[0, :2]).item()
                if dist_to_target < 0.15 and obj_pos[0, 2].item() < 0.06:
                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs,
                        ep_actions,
                        ep_rewards,
                    )
                    collected_count += 1
                    print(f"  [✓] Successfully collected Demo #{collected_count-1} in {step} steps! (dist={dist_to_target:.3f}m)")
                    break
                else:
                    print(f"  [X] Object not on target (dist={dist_to_target:.3f}m, z={obj_pos[0, 2]:.3f}m). Discarding...")
                    break

            if terminated.item() or truncated.item():
                print(f"  [X] Episode terminated early at step {step} (phase={PHASE_NAMES[phase]}). Discarding...")
                break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Scripted Demo Generation Finished!")
    print(f" Total Demos in Dataset : {collected_count}")
    print(f" Time Elapsed           : {elapsed:.1f} seconds ({elapsed/max(1, collected_count-existing_demos):.1f} s/demo)")
    print(f" Saved Dataset          : {args_cli.dataset_file}")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
