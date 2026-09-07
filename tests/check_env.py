# Quick diagnostic script to inspect robot joint order and action mapping
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from dual_arm_il.configs.env_cfg import DualArmILEnvCfg

env_cfg = DualArmILEnvCfg()
env_cfg.sim.device = args_cli.device
env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

robot = env.scene["robot"]
print("\n" + "=" * 60)
print("--- ALL ROBOT JOINTS ---")
for i, name in enumerate(robot.data.joint_names):
    print(f"  Joint {i:2d}: {name}")

print("\n--- ALL ROBOT BODIES ---")
for i, name in enumerate(robot.data.body_names):
    print(f"  Body {i:2d}: {name}")

print("\n--- ACTION MANAGER TERMS ---")
for term_name in env.action_manager.active_terms:
    term = env.action_manager._terms[term_name]
    print(f"\nTerm: {term_name} (class: {type(term).__name__})")
    print(f"  Action Dim: {term.action_dim}")
    if hasattr(term, "_joint_names"):
        print(f"  Joint Names ({len(term._joint_names)}): {term._joint_names}")
    if hasattr(term, "_joint_ids"):
        print(f"  Joint IDs: {term._joint_ids}")
    if hasattr(term, "_offset"):
        print(f"  Offset: {term._offset}")
    if hasattr(term, "_scale"):
        print(f"  Scale: {term._scale}")

print("=" * 60 + "\n")

env.close()
simulation_app.close()
