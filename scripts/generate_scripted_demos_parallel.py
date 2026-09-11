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
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
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
    damping: torch.Tensor | float = 0.05,
    q_current: torch.Tensor | None = None,
    q_nominal: torch.Tensor | None = None,
    k_null: torch.Tensor | float = 0.5,
) -> torch.Tensor:
    # Vectorized IK solver supporting N environments
    N = jacobian.shape[0]
    j = jacobian # (N, 6, 7)
    e = delta_pose.unsqueeze(-1) # (N, 6, 1)
    
    jjt = torch.bmm(j, j.transpose(1, 2)) # (N, 6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype).unsqueeze(0).expand(N, 6, 6)
    
    if isinstance(damping, torch.Tensor):
        damping_term = (damping ** 2).unsqueeze(-1).unsqueeze(-1) * identity
    else:
        damping_term = (damping ** 2) * identity
        
    inv_term = torch.inverse(jjt + damping_term)
    j_pinv = torch.bmm(j.transpose(1, 2), inv_term) # (N, 7, 6)
    
    delta_q = torch.bmm(j_pinv, e).squeeze(-1) # (N, 7)
    
    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[2], device=j.device, dtype=j.dtype).unsqueeze(0).expand(N, 7, 7)
        null_proj = eye_n - torch.bmm(j_pinv, j) # (N, 7, 7)
        q_err = (q_nominal - q_current).unsqueeze(-1) # (N, 7, 1)
        
        if isinstance(k_null, torch.Tensor):
            k_null_term = k_null.unsqueeze(-1).unsqueeze(-1)
        else:
            k_null_term = k_null
            
        delta_q_null = torch.bmm(null_proj, k_null_term * q_err).squeeze(-1)
        delta_q = delta_q + delta_q_null

    return delta_q # (N, 7)



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


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, init_states: dict = None, obj_traj: list = None, joint_traj: list = None):
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
        if init_states:
            for k, v in init_states.items():
                demo_group.create_dataset(k, data=v)
                
        if obj_traj is not None:
            demo_group.create_dataset("object_poses", data=np.array(obj_traj, dtype=np.float32), compression="gzip")
        if joint_traj is not None:
            demo_group.create_dataset("robot_joint_poses", data=np.array(joint_traj, dtype=np.float32), compression="gzip")

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
    cfg.scene.num_envs = args_cli.num_envs

    print(f"[Scripted Demo] Initializing Isaac Lab Environment with {cfg.scene.num_envs} envs...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target"]

    num_envs = env.num_envs

    # Body Indices
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Joint Indices in Articulation
    left_arm_joint_ids, left_arm_names = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, right_arm_names = robot.find_joints("panda_joint[1-7]_0")

    # Gripper finger joint indices
    left_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]$")
    right_gripper_joint_ids, _ = robot.find_joints("panda_finger_joint[1-2]_0")

    # Safe standby postures
    left_standby_joints = robot.data.default_joint_pos[:, left_arm_joint_ids].clone()
    left_standby_joints[:, 0] = 0.60
    right_standby_joints = robot.data.default_joint_pos[:, right_arm_joint_ids].clone()
    right_standby_joints[:, 0] = -0.60

    right_standby_joints[:, 4] += 0.5
    left_standby_joints[:, 4] -= 0.5

    q_nominal_left = left_standby_joints.clone()
    q_nominal_right = right_standby_joints.clone()

    fixed_z_down = torch.tensor([[0.0, 0.0, -1.0]], device=args_cli.device).repeat(num_envs, 1)
    fixed_x_along = torch.tensor([[1.0, 0.0, 0.0]], device=args_cli.device).repeat(num_envs, 1)

    num_base_dofs = getattr(robot, "num_base_dofs", 0)
    jacobi_right_joint_ids = [j + num_base_dofs for j in right_arm_joint_ids]
    jacobi_left_joint_ids = [j + num_base_dofs for j in left_arm_joint_ids]

    arm_term = env.action_manager._terms["arm_action"]
    print(f"[Scripted Demo] Action Manager 'arm_action' controls {len(arm_term._joint_ids)} joints.")
    print(f"  Scale : {arm_term._scale}")

    soft_lower = robot.data.soft_joint_pos_limits[:, :, 0]
    soft_upper = robot.data.soft_joint_pos_limits[:, :, 1]

    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])

    print(f"[Scripted Demo] Existing demonstrations: {existing_demos}")
    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    obs, _ = env.reset()

    # Per-env state
    phase = torch.full((num_envs,), PHASE_INIT, dtype=torch.long, device=args_cli.device)
    phase_timer = torch.zeros((num_envs,), dtype=torch.long, device=args_cli.device)
    
    latched_align_sign_set = torch.zeros((num_envs,), dtype=torch.bool, device=args_cli.device)
    latched_align_sign = torch.zeros((num_envs, 1), dtype=torch.float32, device=args_cli.device)
    latched_cube_z = torch.zeros((num_envs, 3), dtype=torch.float32, device=args_cli.device)
    target_cube_z = torch.zeros((num_envs, 3), dtype=torch.float32, device=args_cli.device)
    target_cube_z_for_wrist = torch.zeros((num_envs, 3), dtype=torch.float32, device=args_cli.device)
    latched_left_grasp_offset = torch.zeros((num_envs, 3), dtype=torch.float32, device=args_cli.device)
    
    commanded_left = left_standby_joints.clone()
    commanded_right = right_standby_joints.clone()
    
    left_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)
    right_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)

    ep_obs = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_obj_traj = [[] for _ in range(num_envs)]
    ep_joint_traj = [[] for _ in range(num_envs)]
    init_states = [{} for _ in range(num_envs)]

    def reset_env_state(env_idx, full_reset=True):
        phase[env_idx] = PHASE_INIT
        phase_timer[env_idx] = 0
        latched_align_sign_set[env_idx] = False
        commanded_left[env_idx] = left_standby_joints[0]
        commanded_right[env_idx] = right_standby_joints[0]
        left_gripper_cmd[env_idx] = 1.0
        right_gripper_cmd[env_idx] = 1.0
        
        if full_reset:
            ep_obs[env_idx].clear()
            ep_actions[env_idx].clear()
            ep_rewards[env_idx].clear()
            ep_obj_traj[env_idx].clear()
            ep_joint_traj[env_idx].clear()
            
            init_states[env_idx] = {
                "init_object_pos": (obj.data.root_pos_w[env_idx] - env.scene.env_origins[env_idx]).clone().cpu().numpy(),
                "init_object_quat": obj.data.root_quat_w[env_idx].clone().cpu().numpy(),
                "init_target_pos": (target.data.root_pos_w[env_idx] - env.scene.env_origins[env_idx]).clone().cpu().numpy(),
                "init_target_quat": target.data.root_quat_w[env_idx].clone().cpu().numpy(),
                "init_robot_pos": (robot.data.root_pos_w[env_idx] - env.scene.env_origins[env_idx]).clone().cpu().numpy(),
                "init_robot_quat": robot.data.root_quat_w[env_idx].clone().cpu().numpy(),
                "init_robot_joint_pos": robot.data.joint_pos[env_idx].clone().cpu().numpy(),
                "init_robot_joint_vel": robot.data.joint_vel[env_idx].clone().cpu().numpy(),
            }

    for i in range(num_envs):
        reset_env_state(i)

    start_time = time.time()
    
    print(f"\n>>> Generating Scripted Demos (Target: {target_count}) <<<")

    step_count = 0
    while simulation_app.is_running() and collected_count < target_count:
        step_count += 1
        
        obj_pos_all = obj.data.root_pos_w.clone()
        obj_quat_all = obj.data.root_quat_w.clone()
        target_pos_all = target.data.root_pos_w.clone()

        wrist_pos_r_all = robot.data.body_pos_w[:, right_hand_idx]
        wrist_quat_r_all = robot.data.body_quat_w[:, right_hand_idx]
        tcp_pos_r_all, z_dir_r_all = compute_tcp(wrist_pos_r_all, wrist_quat_r_all)

        wrist_pos_l_all = robot.data.body_pos_w[:, left_hand_idx]
        wrist_quat_l_all = robot.data.body_quat_w[:, left_hand_idx]
        tcp_pos_l_all, z_dir_l_all = compute_tcp(wrist_pos_l_all, wrist_quat_l_all)

        right_gripper_width_all = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1)
        left_gripper_width_all = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1)

        ow, ox, oy, oz = obj_quat_all[:, 0], obj_quat_all[:, 1], obj_quat_all[:, 2], obj_quat_all[:, 3]
        cube_z_all = torch.stack([
            2.0 * (ox * oz + ow * oy),
            2.0 * (oy * oz - ow * ox),
            1.0 - 2.0 * (ox * ox + oy * oy),
        ], dim=-1)

        delta_r_pos_all = torch.zeros((num_envs, 3), device=args_cli.device)
        delta_r_rot_all = torch.zeros((num_envs, 3), device=args_cli.device)
        delta_l_pos_all = torch.zeros((num_envs, 3), device=args_cli.device)
        delta_l_rot_all = torch.zeros((num_envs, 3), device=args_cli.device)
        
        k_null_r_all = torch.full((num_envs,), 0.05, device=args_cli.device)
        k_null_l_all = torch.full((num_envs,), 0.05, device=args_cli.device)
        damping_r_all = torch.full((num_envs,), 0.05, device=args_cli.device)
        damping_l_all = torch.full((num_envs,), 0.05, device=args_cli.device)
        
        for i in range(num_envs):
            if collected_count >= target_count:
                break
                
            p = phase[i].item()
            phase_timer[i] += 1
            ptimer = phase_timer[i].item()
            
            obj_pos = obj_pos_all[i:i+1]
            obj_quat = obj_quat_all[i:i+1]
            target_pos = target_pos_all[i:i+1]
            cube_z = cube_z_all[i:i+1]
            
            wrist_pos_r = wrist_pos_r_all[i:i+1]
            wrist_quat_r = wrist_quat_r_all[i:i+1]
            tcp_pos_r = tcp_pos_r_all[i:i+1]
            
            wrist_pos_l = wrist_pos_l_all[i:i+1]
            wrist_quat_l = wrist_quat_l_all[i:i+1]
            tcp_pos_l = tcp_pos_l_all[i:i+1]
            
            right_gripper_width = right_gripper_width_all[i].item()
            left_gripper_width = left_gripper_width_all[i].item()
            
            env_origin = env.scene.env_origins[i:i+1]
            
            # Fail-fast
            if 5 <= p <= 11 and (obj_pos[0, 2] - env_origin[0, 2]).item() < 0.05:
                print(f"  [X] Env {i}: Dropped baton in mid-air at phase {PHASE_NAMES[p]}! Discarding...")
                reset_env_state(i, full_reset=True)
                continue

            if not latched_align_sign_set[i]:
                latched_align_sign[i:i+1] = torch.sign(cube_z[:, 1:2] + 1e-6)
                latched_cube_z[i:i+1] = cube_z
                target_cube_z[i:i+1] = torch.tensor([[0.0, latched_align_sign[i].item(), 0.0]], device=args_cli.device)
                latched_align_sign_set[i] = True
                
            l_sign = latched_align_sign[i:i+1]
            l_cube_z = latched_cube_z[i:i+1]
            t_cube_z = target_cube_z[i:i+1]
            
            pick_target = obj_pos - l_sign * 0.055 * l_cube_z
            place_target = obj_pos + l_sign * 0.055 * t_cube_z
            
            handover_pos = env_origin + torch.tensor([[0.30, 0.0, 0.20]], device=args_cli.device)
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
            
            # STATE MACHINE
            if p == PHASE_INIT:
                right_gripper_cmd[i] = 1.0
                left_gripper_cmd[i] = 1.0
                commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                if ptimer > 15:
                    phase[i] = PHASE_RIGHT_HOVER
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_HOVER:
                commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                hover_tgt = pick_target.clone()
                hover_tgt[:, 2] = pick_target[:, 2] + 0.08
                err = hover_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(hover_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(hover_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, l_sign * l_cube_z, directed=True)
                if (dist_xy < 0.025 and dist_z < 0.03) or ptimer > 90:
                    phase[i] = PHASE_RIGHT_DESCEND
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_DESCEND:
                commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                descend_tgt = pick_target.clone()
                descend_tgt[:, 2] = torch.clamp(pick_target[:, 2], min=env_origin[0, 2] + 0.025)
                err = descend_tgt - tcp_pos_r
                dist = torch.norm(err)
                dist_xy = torch.norm(descend_tgt[:, :2] - tcp_pos_r[:, :2])
                dist_z = torch.abs(descend_tgt[:, 2] - tcp_pos_r[:, 2])
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, l_sign * l_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                reached = (dist_xy < 0.010 and dist_z < 0.005)
                if reached or ptimer > 160:
                    phase[i] = PHASE_RIGHT_GRASP
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_GRASP:
                commanded_left[i:i+1] = commanded_left[i:i+1] + torch.clamp(left_standby_joints[i:i+1] - commanded_left[i:i+1], min=-0.04, max=0.04)
                right_gripper_cmd[i] = -1.0
                if ptimer > 25:
                    if right_gripper_width > 0.025:
                        w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                        x_curr = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                        target_cube_z_for_wrist[i:i+1] = l_sign * t_cube_z
                        phase[i] = PHASE_RIGHT_LIFT
                        phase_timer[i] = 0
                    else:
                        right_gripper_cmd[i] = 1.0
                        if ptimer > 40:
                            phase[i] = PHASE_RIGHT_DESCEND
                            phase_timer[i] = 0

            elif p == PHASE_RIGHT_LIFT:
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(t_cube_z[:, 1:2] + 1e-6) * t_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                
                lift_tgt = pick_target.clone()
                lift_tgt[:, 2] = env_origin[0, 2] + 0.18
                err = lift_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, l_sign * l_cube_z, gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.025 or ptimer > 70:
                    phase[i] = PHASE_RIGHT_HANDOVER
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_HANDOVER:
                k_null_r = 0.0
                damping_r = 0.25
                err = handover_pos - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                t_wrist = target_cube_z_for_wrist[i:i+1]
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, t_wrist, gain=0.1, max_rot_step=0.020, directed=True)

                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(t_cube_z[:, 1:2] + 1e-6) * t_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)

                w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
                x_curr_r = torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=-1)
                rot_aligned = (1.0 - torch.sum(x_curr_r * t_wrist, dim=-1)) < 0.005
                
                if (dist < 0.03 and rot_aligned.item()) or ptimer > 400:
                    phase[i] = PHASE_LEFT_APPROACH
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_APPROACH:
                k_null_r = 0.05
                right_gripper_cmd[i] = -1.0
                
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
                left_orient_target = -torch.sign(t_cube_z[:, 1:2] + 1e-6) * t_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)
                
                reached_l = (dist_xy < 0.025 and dist_z < 0.015)
                if reached_l or ptimer > 160:
                    phase[i] = PHASE_LEFT_GRASP
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_GRASP:
                k_null_r = 0.05
                right_gripper_cmd[i] = -1.0
                left_gripper_cmd[i] = -1.0
                if ptimer > 35:
                    if left_gripper_width > 0.025:
                        latched_left_grasp_offset[i:i+1] = tcp_pos_l - obj_pos
                        phase[i] = PHASE_RIGHT_RELEASE
                        phase_timer[i] = 0
                    else:
                        left_gripper_cmd[i] = 1.0
                        if ptimer > 50:
                            phase[i] = PHASE_LEFT_APPROACH
                            phase_timer[i] = 0

            elif p == PHASE_RIGHT_RELEASE:
                k_null_r = 0.05
                left_gripper_cmd[i] = -1.0
                right_gripper_cmd[i] = 1.0
                if ptimer > 15:
                    phase[i] = PHASE_RIGHT_LIFT_CLEAR
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_LIFT_CLEAR:
                k_null_r = 0.05
                left_gripper_cmd[i] = -1.0
                right_gripper_cmd[i] = 1.0
                
                lift_clear_tgt = handover_pos.clone()
                lift_clear_tgt[:, 1] -= 0.15
                lift_clear_tgt[:, 2] = env_origin[0, 2] + 0.35
                err = lift_clear_tgt - tcp_pos_r
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_r_pos = (err / (dist + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, target_cube_z_for_wrist[i:i+1], gain=0.2, max_rot_step=0.020, directed=True)
                if dist < 0.03 or ptimer > 60:
                    phase[i] = PHASE_RIGHT_RETREAT
                    phase_timer[i] = 0

            elif p == PHASE_RIGHT_RETREAT:
                lift_clear_tgt = handover_pos.clone()
                lift_clear_tgt[:, 1] -= 0.15
                lift_clear_tgt[:, 2] = env_origin[0, 2] + 0.35
                retreat_tgt = lift_clear_tgt.clone()
                retreat_tgt[:, 0] -= 0.15
                err = retreat_tgt - tcp_pos_r
                step_size = min(0.012, torch.norm(err).item())
                delta_r_pos = (err / (torch.norm(err) + 1e-6)) * step_size
                delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, t_cube_z)
                
                left_gripper_cmd[i] = -1.0
                right_gripper_cmd[i] = 1.0
                commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                if ptimer > 15:
                    phase[i] = PHASE_LEFT_HOVER_TARGET
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_HOVER_TARGET:
                commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset[i:i+1] + manual_offset
                tgt_hover = final_place_target.clone()
                tgt_hover[:, 2] = target_pos[:, 2] + 0.12
                err = tgt_hover - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.016, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(t_cube_z[:, 1:2] + 1e-6) * t_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                dist_xy = torch.norm(tgt_hover[:, :2] - tcp_pos_l[:, :2])
                dist_z = torch.abs(tgt_hover[:, 2] - tcp_pos_l[:, 2])
                if (dist_xy < 0.01 and dist_z < 0.015) or ptimer > 70:
                    phase[i] = PHASE_LEFT_LOWER_TARGET
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_LOWER_TARGET:
                commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                manual_offset = torch.tensor([[-0.08, -0.08, 0.0]], device=args_cli.device)
                final_place_target = target_pos + latched_left_grasp_offset[i:i+1] + manual_offset
                tgt_lower = final_place_target.clone()
                tgt_lower[:, 2] = env_origin[0, 2] + 0.022
                err = tgt_lower - tcp_pos_l
                dist = torch.norm(err)
                dist_z = torch.abs(tgt_lower[:, 2] - tcp_pos_l[:, 2])
                step_size = min(0.008, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                left_orient_target = -torch.sign(t_cube_z[:, 1:2] + 1e-6) * t_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.10, max_rot_step=0.010, directed=True)
                if (dist < 0.015 or dist_z < 0.006) or ptimer > 60:
                    phase[i] = PHASE_LEFT_RELEASE
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_RELEASE:
                commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                left_gripper_cmd[i] = 1.0
                if ptimer > 10:
                    phase[i] = PHASE_LEFT_RETREAT
                    phase_timer[i] = 0

            elif p == PHASE_LEFT_RETREAT:
                commanded_right[i:i+1] = commanded_right[i:i+1] + torch.clamp(right_standby_joints[i:i+1] - commanded_right[i:i+1], min=-0.04, max=0.04)
                left_gripper_cmd[i] = 1.0
                tgt_up = target_pos.clone()
                tgt_up[:, 2] += 0.20
                err = tgt_up - tcp_pos_l
                dist = torch.norm(err)
                step_size = min(0.012, dist.item())
                delta_l_pos = (err / (dist + 1e-6)) * step_size
                if dist < 0.03 or ptimer > 40:
                    phase[i] = PHASE_SUCCESS
                    phase_timer[i] = 0

            delta_r_pos_all[i:i+1] = delta_r_pos
            delta_r_rot_all[i:i+1] = delta_r_rot
            delta_l_pos_all[i:i+1] = delta_l_pos
            delta_l_rot_all[i:i+1] = delta_l_rot
            
            k_null_r_all[i] = k_null_r
            k_null_l_all[i] = k_null_l
            damping_r_all[i] = damping_r
            damping_l_all[i] = damping_l

        # Batched IK for all envs
        jacobians = robot.root_physx_view.get_jacobians()
        
        delta_r_pose = torch.cat([delta_r_pos_all, delta_r_rot_all], dim=-1)
        j_r_wrist = jacobians[:, jacobi_right_hand_idx, :6, :][:, :, jacobi_right_joint_ids]
        j_r_tcp = get_tcp_jacobian(j_r_wrist, wrist_pos_r_all, tcp_pos_r_all)
        dq_r = solve_pose_dls_ik(
            j_r_tcp,
            delta_r_pose,
            damping=damping_r_all,
            q_current=commanded_right,
            q_nominal=q_nominal_right,
            k_null=k_null_r_all,
        )
        dq_r = torch.clamp(dq_r, min=-0.08, max=0.08)
        commanded_right += dq_r
        commanded_right = torch.clamp(
            commanded_right,
            min=soft_lower[:, right_arm_joint_ids] + 0.02,
            max=soft_upper[:, right_arm_joint_ids] - 0.02,
        )

        delta_l_pose = torch.cat([delta_l_pos_all, delta_l_rot_all], dim=-1)
        j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
        j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l_all, tcp_pos_l_all)
        
        for i in range(num_envs):
            if phase[i] == PHASE_LEFT_HOVER_TARGET:
                j_l_tcp[i:i+1, 3:6, :] = 0.0
                
        dq_l = solve_pose_dls_ik(
            j_l_tcp,
            delta_l_pose,
            damping=damping_l_all,
            q_current=commanded_left,
            q_nominal=q_nominal_left,
            k_null=k_null_l_all,
        )
        dq_l = torch.clamp(dq_l, min=-0.08, max=0.08)
        commanded_left += dq_l
        commanded_left = torch.clamp(
            commanded_left,
            min=soft_lower[:, left_arm_joint_ids] + 0.02,
            max=soft_upper[:, left_arm_joint_ids] - 0.02,
        )

        # Action creation
        target_all_joints = robot.data.default_joint_pos.clone()
        target_all_joints[:, left_arm_joint_ids] = commanded_left
        target_all_joints[:, right_arm_joint_ids] = commanded_right
        arm_targets = target_all_joints[:, arm_term._joint_ids]
        raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale
        action = torch.cat([raw_arm_action, left_gripper_cmd, right_gripper_cmd], dim=-1)

        # Step sim
        policy_obs_all = obs["policy"].detach().cpu().numpy()
        action_np_all = action.detach().cpu().numpy()
        
        for i in range(num_envs):
            ep_obs[i].append(policy_obs_all[i])
            ep_actions[i].append(action_np_all[i])
            
            obj_pose = torch.cat([env.scene["object"].data.root_pos_w[i:i+1] - env.scene.env_origins[i:i+1], env.scene["object"].data.root_quat_w[i:i+1]], dim=-1).squeeze(0).cpu().numpy()
            ep_obj_traj[i].append(obj_pose)
            ep_joint_traj[i].append(robot.data.joint_pos[i].cpu().numpy())

        obs, reward, terminated, truncated, _ = env.step(action)
        reward_np = reward.detach().cpu().numpy()
        
        for i in range(num_envs):
            ep_rewards[i].append(float(reward_np[i]))

        # Termination & Success logic
        for i in range(num_envs):
            if collected_count >= target_count:
                break
                
            if phase[i] == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos_all[i, :2] - target_pos_all[i, :2]).item()
                if dist_to_target < 0.15 and (obj_pos_all[i, 2] - env.scene.env_origins[i, 2]).item() < 0.06:
                    save_episode_to_hdf5(
                        args_cli.dataset_file,
                        collected_count,
                        ep_obs[i],
                        ep_actions[i],
                        ep_rewards[i],
                        init_states[i],
                        ep_obj_traj[i],
                        ep_joint_traj[i],
                    )
                    collected_count += 1
                    print(f"  [✓] Env {i}: Collected Demo #{collected_count-1} in {len(ep_actions[i])} steps! (dist={dist_to_target:.3f}m)")
                else:
                    print(f"  [X] Env {i}: Object not on target (dist={dist_to_target:.3f}m). Discarding...")
                
                reset_env_state(i, full_reset=True)
                
            elif terminated[i].item() or truncated[i].item():
                print(f"  [X] Env {i}: Episode terminated early at phase {PHASE_NAMES[phase[i]]}. Discarding...")
                reset_env_state(i, full_reset=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Scripted Demo Generation Finished!")
    print(f" Total Demos in Dataset : {collected_count}")
    print(f" Time Elapsed           : {elapsed:.1f} seconds")
    print(f" Saved Dataset          : {args_cli.dataset_file}")
    print("=" * 60)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
