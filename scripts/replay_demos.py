# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay Demonstrations Script.

Replays recorded demonstration trajectories in Isaac Sim to visually
inspect and verify motion quality and task success.

Usage:
    python scripts/replay_demos.py --dataset=data/demos.hdf5 --demo_idx=0
"""

import argparse
import os
import sys
import time
import h5py
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Replay recorded demonstrations.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--demo_idx", type=int, default=0, help="Demo index to replay (or -1 for all).")
parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds.")
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


def replay_single_demo(env: ManagerBasedRLEnv, actions: list, demo_name: str, delay: float, init_states: dict = None):
    print(f"\n--- Replaying {demo_name} ({len(actions)} steps) ---")
    env.reset()

    if init_states:
        # Override object and target initial states
        if "init_object_pos" in init_states:
            obj_state = env.scene["object"].data.default_root_state.clone()
            obj_state[:, :3] = torch.tensor(init_states["init_object_pos"], device=env.device)
            obj_state[:, 3:7] = torch.tensor(init_states["init_object_quat"], device=env.device)
            env.scene["object"].write_root_state_to_sim(obj_state)
            
        if "init_target_pos" in init_states:
            tgt_state = env.scene["target"].data.default_root_state.clone()
            tgt_state[:, :3] = torch.tensor(init_states["init_target_pos"], device=env.device)
            tgt_state[:, 3:7] = torch.tensor(init_states["init_target_quat"], device=env.device)
            env.scene["target"].write_root_state_to_sim(tgt_state)

    for step_idx, act in enumerate(actions):
        if not simulation_app.is_running():
            break

        act_t = torch.tensor(act, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        env.step(act_t)

        if delay > 0:
            time.sleep(delay)

    print(f"Finished {demo_name}.")


def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            env.close()
            simulation_app.close()
            return

        if args_cli.demo_idx >= 0:
            target_key = f"demo_{args_cli.demo_idx}"
            if target_key not in demo_keys:
                print(f"[Replay] Demo index {args_cli.demo_idx} not found. Available: {demo_keys}")
                env.close()
                simulation_app.close()
                return
            target_demos = [target_key]
        else:
            target_demos = demo_keys

        for key in target_demos:
            actions = data_grp[key]["actions"][:]
            init_states = None
            if "init_object_pos" in data_grp[key]:
                init_states = {
                    "init_object_pos": data_grp[key]["init_object_pos"][:],
                    "init_object_quat": data_grp[key]["init_object_quat"][:],
                    "init_target_pos": data_grp[key]["init_target_pos"][:],
                    "init_target_quat": data_grp[key]["init_target_quat"][:],
                }
            replay_single_demo(env, actions, key, args_cli.delay, init_states)
            time.sleep(1.0)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
