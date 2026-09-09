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

parser = argparse.ArgumentParser(description="Replay multiple demonstrations in parallel.")
parser.add_argument(
    "--dataset",
    type=str,
    default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
    help="Path to HDF5 dataset file.",
)
parser.add_argument("--num_parallel", type=int, default=4, help="Number of demos to play simultaneously.")
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

def main():
    if not os.path.exists(args_cli.dataset):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset}")

    with h5py.File(args_cli.dataset, "r") as f:
        data_grp = f["data"]
        demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")], key=lambda x: int(x.split("_")[1]))

        if len(demo_keys) == 0:
            print("[Replay] No demonstrations found in dataset!")
            simulation_app.close()
            return
            
        num_parallel = min(args_cli.num_parallel, len(demo_keys))
        target_demos = demo_keys[:num_parallel]
        print(f"[Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Load all actions for the selected demos
        all_actions = []
        all_init_states = []
        max_length = 0
        for key in target_demos:
            acts = data_grp[key]["actions"][:]
            all_actions.append(acts)
            if len(acts) > max_length:
                max_length = len(acts)
            if "init_object_pos" in data_grp[key]:
                state_dict = {
                    "init_object_pos": data_grp[key]["init_object_pos"][:],
                    "init_object_quat": data_grp[key]["init_object_quat"][:],
                    "init_target_pos": data_grp[key]["init_target_pos"][:],
                    "init_target_quat": data_grp[key]["init_target_quat"][:],
                }
                if "init_robot_pos" in data_grp[key]:
                    state_dict.update({
                        "init_robot_pos": data_grp[key]["init_robot_pos"][:],
                        "init_robot_quat": data_grp[key]["init_robot_quat"][:],
                        "init_robot_joint_pos": data_grp[key]["init_robot_joint_pos"][:],
                        "init_robot_joint_vel": data_grp[key]["init_robot_joint_vel"][:],
                    })
                all_init_states.append(state_dict)
            else:
                all_init_states.append(None)

    # Setup environment with num_envs = num_parallel
    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = num_parallel
    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DualArmILEnvCfg},
        disable_env_checker=True,
    )
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped

    env.reset()
    
    # Override initial states for each parallel environment
    if any(s is not None for s in all_init_states):
        obj_state = env.scene["object"].data.default_root_state.clone()
        tgt_state = env.scene["target"].data.default_root_state.clone()
        
        for i, s in enumerate(all_init_states):
            if s is not None:
                obj_state[i, :3] = torch.tensor(s["init_object_pos"], device=env.device)
                obj_state[i, 3:7] = torch.tensor(s["init_object_quat"], device=env.device)
                tgt_state[i, :3] = torch.tensor(s["init_target_pos"], device=env.device)
                tgt_state[i, 3:7] = torch.tensor(s["init_target_quat"], device=env.device)
                
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["target"].write_root_state_to_sim(tgt_state)
        
        # Override robot states
        if "init_robot_pos" in all_init_states[0]:
            rob_state = env.scene["robot"].data.default_root_state.clone()
            j_pos = env.scene["robot"].data.default_joint_pos.clone()
            j_vel = env.scene["robot"].data.default_joint_vel.clone()
            
            for i, s in enumerate(all_init_states):
                if s is not None and "init_robot_pos" in s:
                    rob_state[i, :3] = torch.tensor(s["init_robot_pos"], device=env.device)
                    rob_state[i, 3:7] = torch.tensor(s["init_robot_quat"], device=env.device)
                    j_pos[i] = torch.tensor(s["init_robot_joint_pos"], device=env.device)
                    j_vel[i] = torch.tensor(s["init_robot_joint_vel"], device=env.device)
                    
            env.scene["robot"].write_root_state_to_sim(rob_state)
            env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)

    print(f"\n--- Starting parallel replay (Max steps: {max_length}) ---")
    
    # Run loop
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        # Construct batched action [num_envs, act_dim]
        # If a demo has finished, repeat its last action
        batched_actions = []
        for i in range(num_parallel):
            acts = all_actions[i]
            idx = min(step_idx, len(acts) - 1)
            batched_actions.append(acts[idx])
            
        act_t = torch.tensor(batched_actions, dtype=torch.float32, device=args_cli.device)
        env.step(act_t)

        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished parallel replay.")
    time.sleep(2.0)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
