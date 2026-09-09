# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration Collection Script for Dual Arm via Teleoperation.

Collects expert demonstrations in Isaac Sim using the DualArmTeleopController,
and exports successful trajectories into an HDF5 dataset compatible with
BC, Diffusion Policy, and ACT.

Usage:
    # Run from the isaac_lab root or dual_arm_il directory:
    python /home/optimus/isaac_lab/dual_arm_il/teleop/collect_demos.py --num_demos=20
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

# Isaac Lab AppLauncher (Must be called before importing omni or isaaclab modules)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect dual arm demonstrations via teleoperation.")
parser.add_argument("--num_demos", type=int, default=20, help="Target number of demonstrations to collect.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to the output HDF5 dataset file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after simulation app launch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
try:
    from dual_arm_il.configs.env_cfg import DualArmILEnvCfg
    from dual_arm_il.teleop.dual_arm_teleop import DualArmTeleopController
except ModuleNotFoundError:
    from configs.env_cfg import DualArmILEnvCfg
    from teleop.dual_arm_teleop import DualArmTeleopController


def solve_dls_ik(
    jacobian: torch.Tensor,
    delta_pose: torch.Tensor,
    damping: float = 0.05,
) -> torch.Tensor:
    """Damped Least Squares (DLS) Inverse Kinematics.

    Args:
        jacobian: Jacobian matrix of shape (1, 6, 7).
        delta_pose: Desired task-space delta [dx, dy, dz, droll, dpitch, dyaw] of shape (1, 6).
        damping: Damping factor lambda.

    Returns:
        delta_q: Change in joint angles of shape (1, 7).
    """
    # J: (1, 6, 7)
    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)

    # J * J^T + lambda^2 * I
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)  # (6, 6)

    # J^T * inv * e
    delta_q = torch.matmul(j.transpose(0, 1), torch.matmul(inv_term, e))  # (7,)
    return delta_q.unsqueeze(0)


def save_episode_to_hdf5(hdf5_path: str, ep_idx: int, observations: list, actions: list, rewards: list, init_states: dict = None, obj_traj: list = None, joint_traj: list = None):
    """Save a single successful demonstration to the HDF5 file."""
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

        # Update total samples attribute in data group
        total_samples = f["data"].attrs.get("total", 0) + len(act_array)
        f["data"].attrs["total"] = total_samples

    print(f"[Dataset] Successfully saved demo_{ep_idx} ({len(act_array)} steps) to {hdf5_path}")


def main():
    cfg = DualArmILEnvCfg()
    cfg.sim.device = args_cli.device

    print("[Collect] Initializing Isaac Lab environment...")
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=cfg).unwrapped
    robot = env.scene["robot"]

    # Locate body indices and joint indices
    # Right arm: "panda_hand_0", joints "panda_joint[1-7]_0"
    # Left arm:  "panda_hand",   joints "panda_joint[1-7]"
    right_hand_idx = robot.find_bodies("panda_hand_0")[0][0]
    left_hand_idx = robot.find_bodies("panda_hand$")[0][0]

    # Find 14 arm joint indices in articulation
    left_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]$")
    right_arm_joint_ids, _ = robot.find_joints("panda_joint[1-7]_0")

    print(f"[Collect] Right hand body index: {right_hand_idx}, Left hand body index: {left_hand_idx}")
    print(f"[Collect] Left arm joints: {left_arm_joint_ids}")
    print(f"[Collect] Right arm joints: {right_arm_joint_ids}")

    # For fixed-base articulation in PhysX, the base link is omitted in Jacobian matrices:
    is_fixed = getattr(robot, "is_fixed_base", True)
    jacobi_right_hand_idx = right_hand_idx - 1 if is_fixed else right_hand_idx
    jacobi_left_hand_idx = left_hand_idx - 1 if is_fixed else left_hand_idx

    # Inspect the Action Manager's arm_action term
    arm_term = env.action_manager._terms["arm_action"]

    # Initialize Teleop Controller
    teleop = DualArmTeleopController(device=args_cli.device)

    # Inspect existing dataset to get starting demo index
    existing_demos = 0
    if os.path.exists(args_cli.dataset_file):
        with h5py.File(args_cli.dataset_file, "r") as f:
            if "data" in f:
                existing_demos = len([k for k in f["data"].keys() if k.startswith("demo_")])
    print(f"[Collect] Existing demonstrations found: {existing_demos}")

    collected_count = existing_demos
    target_count = existing_demos + args_cli.num_demos

    # Main collection loop
    while simulation_app.is_running() and collected_count < target_count:
        obs, _ = env.reset()
        teleop.reset()
        
        init_states = {
            "init_object_pos": env.scene["object"].data.root_pos_w.clone().squeeze(0).cpu().numpy(),
            "init_object_quat": env.scene["object"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_target_pos": env.scene["target"].data.root_pos_w.clone().squeeze(0).cpu().numpy(),
            "init_target_quat": env.scene["target"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_pos": env.scene["robot"].data.root_pos_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_quat": env.scene["robot"].data.root_quat_w.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_pos": env.scene["robot"].data.joint_pos.clone().squeeze(0).cpu().numpy(),
            "init_robot_joint_vel": env.scene["robot"].data.joint_vel.clone().squeeze(0).cpu().numpy(),
        }

        # Commanded joint targets buffer (starts at current joint positions)
        current_joint_pos = robot.data.joint_pos.clone()
        commanded_left_joints = current_joint_pos[:, left_arm_joint_ids].clone()
        commanded_right_joints = current_joint_pos[:, right_arm_joint_ids].clone()

        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_obj_traj = []
        ep_joint_traj = []

        print(f"\n>>> Starting Episode for Demo #{collected_count} (Target: {target_count}) <<<")

        step_idx = 0
        while simulation_app.is_running():
            step_idx += 1

            # 1. Read teleoperation command
            active_arm, dpos, drot, r_grip, l_grip = teleop.get_delta_command()

            # 2. Compute IK if there is a delta movement
            delta_pose = np.concatenate([dpos, drot])  # 6D: [dx, dy, dz, drx, dry, drz]
            has_motion = np.linalg.norm(dpos) > 1e-5 or np.linalg.norm(drot) > 1e-5

            if has_motion:
                delta_pose_t = torch.tensor(delta_pose, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
                jacobians = robot.root_physx_view.get_jacobians()  # (1, num_bodies, 6, num_dofs)

                if active_arm == "right":
                    # Jacobians for right hand w.r.t right arm joints
                    j_right = jacobians[:, jacobi_right_hand_idx, :, :][:, :, right_arm_joint_ids]
                    dq_right = solve_dls_ik(j_right, delta_pose_t, damping=0.08)
                    commanded_right_joints += dq_right
                else:
                    # Jacobians for left hand w.r.t left arm joints
                    j_left = jacobians[:, jacobi_left_hand_idx, :, :][:, :, left_arm_joint_ids]
                    dq_left = solve_dls_ik(j_left, delta_pose_t, damping=0.08)
                    commanded_left_joints += dq_left

            # 3. Assemble 16D action with exact action inversion
            target_all_joints = robot.data.default_joint_pos.clone()
            target_all_joints[:, left_arm_joint_ids] = commanded_left_joints
            target_all_joints[:, right_arm_joint_ids] = commanded_right_joints

            arm_targets = target_all_joints[:, arm_term._joint_ids]
            raw_arm_action = (arm_targets - arm_term._offset) / arm_term._scale

            left_grip_t = torch.tensor([[l_grip]], dtype=torch.float32, device=args_cli.device)
            right_grip_t = torch.tensor([[r_grip]], dtype=torch.float32, device=args_cli.device)

            action = torch.cat([raw_arm_action, left_grip_t, right_grip_t], dim=-1)

            # 4. Record step transition
            policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()
            action_np = action.squeeze(0).detach().cpu().numpy()

            ep_obs.append(policy_obs)
            ep_actions.append(action_np)
            
            import torch
            obj_pose = torch.cat([env.scene["object"].data.root_pos_w, env.scene["object"].data.root_quat_w], dim=-1).squeeze(0).cpu().numpy()
            ep_obj_traj.append(obj_pose)
            ep_joint_traj.append(robot.data.joint_pos.squeeze(0).cpu().numpy())

            # 5. Step simulation
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rewards.append(float(reward.mean().item()))

            # 6. Check workflow flags
            if teleop.flag_save_episode:
                save_episode_to_hdf5(
                    args_cli.dataset_file,
                    collected_count,
                    ep_obs,
                    ep_actions,
                    ep_rewards,
                    init_states,
                    ep_obj_traj,
                    ep_joint_traj,
                )
                collected_count += 1
                teleop.reset_episode_flags()
                break

            if teleop.flag_discard_episode:
                print(f"[Collect] Episode discarded by user. Total retained: {collected_count}")
                teleop.reset_episode_flags()
                break

            if teleop.flag_reset_env or terminated.item() or truncated.item():
                if terminated.item():
                    print("[Collect] Episode terminated (e.g. object dropped). Discarding...")
                teleop.reset_episode_flags()
                break

    print(f"\n[Collect] Finished! Total demonstrations collected: {collected_count}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
