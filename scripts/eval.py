# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop Evaluation Script for Dual Arm Imitation Learning Policies.

Tests trained BC, Diffusion Policy, or ACT models in the Isaac Sim simulation
and logs task success metrics.

Usage:
    python scripts/eval.py --algo=diffusion --num_episodes=10
    python scripts/eval.py --algo=act --num_episodes=10
    python scripts/eval.py --algo=bc --num_episodes=10
"""

import argparse
import collections
import os
import sys
import torch

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

parser = argparse.ArgumentParser(description="Evaluate trained imitation learning policies.")
parser.add_argument("--algo", type=str, default="diffusion", choices=["bc", "diffusion", "act"])
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt).")
parser.add_argument("--stats", type=str, default=None, help="Path to normalization stats.pkl.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of evaluation episodes.")
parser.add_argument("--max_steps_per_ep", type=int, default=700, help="Max steps per episode (~23 seconds at 30Hz).")
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
from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def main():
    algo = args_cli.algo.lower()
    ckpt_path = args_cli.checkpoint or os.path.join(PROJECT_ROOT, "checkpoints", algo, "best_model.pt")
    stats_path = args_cli.stats or os.path.join(PROJECT_ROOT, "checkpoints", algo, "stats.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    print("=" * 60)
    print(f" Evaluating Policy: {algo.upper()}")
    print(f" Checkpoint : {ckpt_path}")
    print(f" Stats      : {stats_path}")
    print("=" * 60)

    # Load Normalization Stats
    stats = DualArmDataset.load_stats(stats_path)
    obs_mean = torch.tensor(stats["obs_mean"], device=args_cli.device, dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], device=args_cli.device, dtype=torch.float32)
    act_mean = torch.tensor(stats["act_mean"], device=args_cli.device, dtype=torch.float32)
    act_std = torch.tensor(stats["act_std"], device=args_cli.device, dtype=torch.float32)

    # Load Checkpoint & Instantiate Model
    checkpoint = torch.load(ckpt_path, map_location=args_cli.device)
    cfg = checkpoint.get("config", {})
    obs_dim = checkpoint.get("obs_dim", len(obs_mean))
    act_dim = checkpoint.get("act_dim", len(act_mean))

    if algo == "bc":
        model = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=cfg.get("hidden_dims", [512, 512, 256]))
    elif algo == "diffusion":
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
        )
    elif algo == "act":
        model = ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            latent_dim=cfg.get("latent_dim", 32),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
        )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(args_cli.device)
    model.eval()

    # Create Environment
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    obs_horizon = cfg.get("obs_horizon", 2)
    act_horizon = cfg.get("act_horizon", 8)

    num_success = 0
    total_episodes = args_cli.num_episodes

    print(f"\nStarting {total_episodes} evaluation episodes...\n")

    for ep_idx in range(1, total_episodes + 1):
        obs, _ = env.reset()
        if hasattr(model, "reset_temporal_ensemble"):
            model.reset_temporal_ensemble()

        # Observation queue for Diffusion Policy
        obs_queue = collections.deque(maxlen=obs_horizon)
        # Action queue for multi-step receding horizon execution
        action_queue = collections.deque()

        ep_success = False

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Extract and normalize observation
            raw_obs = obs["policy"].squeeze(0)  # (obs_dim,)
            norm_obs = (raw_obs - obs_mean) / obs_std

            with torch.no_grad():
                if algo == "bc":
                    norm_action = model(norm_obs)
                    action = norm_action * act_std + act_mean

                elif algo == "diffusion":
                    obs_queue.append(norm_obs)
                    while len(obs_queue) < obs_horizon:
                        obs_queue.append(norm_obs)

                    if len(action_queue) == 0:
                        obs_tensor = torch.stack(list(obs_queue), dim=0).unsqueeze(0)  # (1, To, obs_dim)
                        pred_act_chunk = model.predict_action(obs_tensor, num_inference_steps=15)
                        pred_act_chunk = pred_act_chunk.squeeze(0)  # (Tp, act_dim)

                        # Enqueue first act_horizon actions
                        for a_idx in range(min(act_horizon, len(pred_act_chunk))):
                            act_unnorm = pred_act_chunk[a_idx] * act_std + act_mean
                            action_queue.append(act_unnorm)

                    action = action_queue.popleft()

                elif algo == "act":
                    norm_action = model.predict_action_step(norm_obs, current_step=step)
                    action = norm_action * act_std + act_mean

            # Step environment: action shape (1, act_dim)
            action_step = action.unsqueeze(0)
            obs, reward, terminated, truncated, _ = env.step(action_step)

            # Check task success: baton placed near red target
            obj_pos = env.scene["object"].data.root_pos_w.squeeze(0)
            target_pos = env.scene["target"].data.root_pos_w.squeeze(0)
            dist_to_target = torch.norm(obj_pos[:2] - target_pos[:2]).item()

            if dist_to_target < 0.12 and obj_pos[2].item() < 0.05:
                ep_success = True
                print(f"[Episode {ep_idx}] SUCCESS! Placed at target (dist: {dist_to_target:.3f}m, step: {step})")
                break

            if terminated.item() or truncated.item():
                break

        if ep_success:
            num_success += 1
        else:
            print(f"[Episode {ep_idx}] Failed or timed out.")

    success_rate = (num_success / total_episodes) * 100.0
    print("\n" + "=" * 60)
    print(f" Evaluation Completed: {algo.upper()}")
    print(f" Total Episodes : {total_episodes}")
    print(f" Successes      : {num_success}")
    print(f" Success Rate   : {success_rate:.1f}%")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
