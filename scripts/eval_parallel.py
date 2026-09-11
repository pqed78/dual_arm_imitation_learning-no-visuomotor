import argparse
import collections
import os
import sys
import math
import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
for path in [PROJECT_ROOT, PARENT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

# Parse Arguments
parser = argparse.ArgumentParser(description="Evaluate Imitation Learning Policy in Parallel.")
parser.add_argument("--algo", type=str, required=True, choices=["bc", "diffusion", "act"])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=100, help="Total episodes to evaluate")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_steps_per_ep", type=int, default=1000)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from configs.env_cfg import DualArmILEnvCfg
from models import MLPBCPolicy, DiffusionPolicy, ACTPolicy

def main():
    device = torch.device(args_cli.device)
    algo = args_cli.algo
    ckpt_path = args_cli.checkpoint

    save_dir = os.path.dirname(ckpt_path)
    stats_path = os.path.join(save_dir, "stats.pkl")

    if not os.path.exists(stats_path):
        print(f"[Error] Stats file not found at {stats_path}")
        sys.exit(1)
    
    from dataset.il_dataset import DualArmDataset
    stats = DualArmDataset.load_stats(stats_path)
    obs_mean = torch.tensor(stats["obs_mean"], device=args_cli.device, dtype=torch.float32)
    obs_std = torch.tensor(stats["obs_std"], device=args_cli.device, dtype=torch.float32)
    act_mean = torch.tensor(stats["act_mean"], device=args_cli.device, dtype=torch.float32)
    act_std = torch.tensor(stats["act_std"], device=args_cli.device, dtype=torch.float32)

    print(f"[Model] Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    model.to(device)
    model.eval()

    # Create Parallel Environment
    env_cfg = DualArmILEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-v0", cfg=env_cfg).unwrapped

    obs_horizon = cfg.get("obs_horizon", 2)
    act_horizon = cfg.get("act_horizon", 8)

    num_envs = args_cli.num_envs
    total_episodes = args_cli.num_episodes
    num_batches = math.ceil(total_episodes / num_envs)

    print(f"\nStarting {total_episodes} evaluation episodes across {num_batches} batches (Batch Size: {num_envs})...\n")

    total_success = 0
    total_evaluated = 0

    for batch_idx in range(num_batches):
        print(f"\n=== Batch {batch_idx + 1}/{num_batches} ===")
        obs, _ = env.reset()
        if hasattr(model, "reset_temporal_ensemble"):
            model.reset_temporal_ensemble()

        obs_queue = collections.deque(maxlen=obs_horizon)
        action_queue = collections.deque()

        batch_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        batch_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        last_safe_action = None

        for step in range(args_cli.max_steps_per_ep):
            if not simulation_app.is_running():
                break

            # Extract and normalize observation
            raw_obs = obs["policy"]  # (num_envs, obs_dim)
            norm_obs = (raw_obs - obs_mean) / obs_std

            with torch.no_grad():
                if algo == "bc":
                    norm_action = model(norm_obs)
                    action = norm_action * act_std + act_mean

                elif algo == "diffusion":
                    obs_queue.append(norm_obs)
                    if step == 0:
                        max_val, max_idx = torch.abs(norm_obs[0]).max(dim=0)
                        print(f'Env 0 MAX norm_obs: {max_val.item():.3f} at index {max_idx.item()}')
                        max_val1, max_idx1 = torch.abs(norm_obs[1]).max(dim=0)
                        print(f'Env 1 MAX norm_obs: {max_val1.item():.3f} at index {max_idx1.item()}')
                    while len(obs_queue) < obs_horizon:
                        obs_queue.append(norm_obs)

                    if len(action_queue) == 0:
                        # Stack to (To, num_envs, obs_dim) -> Permute to (num_envs, To, obs_dim)
                        obs_tensor = torch.stack(list(obs_queue), dim=0).permute(1, 0, 2)
                        infer_steps = model.num_train_timesteps
                        pred_act_chunk = model.predict_action(obs_tensor, num_inference_steps=infer_steps, use_ddim=False)
                        
                        for a_idx in range(min(act_horizon, pred_act_chunk.shape[1])):
                            act_unnorm = pred_act_chunk[:, a_idx, :] * act_std + act_mean
                            action_queue.append(act_unnorm)

                    action = action_queue.popleft() # (num_envs, act_dim)

                elif algo == "act":
                    # ACT expects (B, obs_dim)
                    norm_action = model.predict_action_step(norm_obs, current_step=step)
                    action = norm_action * act_std + act_mean

            # Freeze done environments so they don't spin out of control
            if last_safe_action is None:
                last_safe_action = action.clone()
            else:
                last_safe_action[~batch_done] = action[~batch_done].clone()
                action[batch_done] = last_safe_action[batch_done]

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Check task success and failures
            obj_pos = env.scene["object"].data.root_pos_w # (num_envs, 3)
            target_pos = env.scene["target"].data.root_pos_w # (num_envs, 3)
            dist_to_target = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)

            # Success condition
            is_success = (dist_to_target < 0.12) & (obj_pos[:, 2] < 0.05)
            newly_succeeded = is_success & ~batch_done
            
            for env_idx in newly_succeeded.nonzero(as_tuple=True)[0]:
                print(f"  [Env {env_idx.item()}] SUCCESS! (Step: {step})")
            
            batch_success |= newly_succeeded
            batch_done |= is_success

            # Failure conditions
            is_dropped = obj_pos[:, 2] < -0.05
            joint_vel = env.scene["robot"].data.joint_vel
            is_spinning = torch.any(torch.abs(joint_vel) > 10.0, dim=1)

            newly_failed = (is_dropped | is_spinning | terminated | truncated) & ~batch_done
            for env_idx in newly_failed.nonzero(as_tuple=True)[0]:
                print(f"  [Env {env_idx.item()}] FAILED (Dropped or OOD). (Step: {step})")
            
            batch_done |= newly_failed

            if batch_done.all():
                print(f"All environments in batch {batch_idx + 1} finished early at step {step}.")
                break

        # Calculate results for this batch
        valid_envs_in_batch = min(num_envs, total_episodes - total_evaluated)
        total_success += batch_success[:valid_envs_in_batch].sum().item()
        total_evaluated += valid_envs_in_batch

        print(f"Batch {batch_idx + 1} Success Rate: {batch_success[:valid_envs_in_batch].sum().item()} / {valid_envs_in_batch}")

    success_rate = (total_success / total_evaluated) * 100.0
    print("\n" + "=" * 60)
    print(f" Parallel Evaluation Completed: {algo.upper()}")
    print(f" Total Evaluated: {total_evaluated}")
    print(f" Successes      : {total_success}")
    print(f" Success Rate   : {success_rate:.1f}%")
    print("=" * 60)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
