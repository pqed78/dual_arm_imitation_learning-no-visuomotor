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

parser = argparse.ArgumentParser(description="Kinematic Replay of multiple demonstrations (100% Visual Accuracy).")
parser.add_argument("--dataset", type=str, default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"))
parser.add_argument("--num_parallel", type=int, default=4, help="Number of demos to play simultaneously.")
parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames in seconds.")
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
        print(f"[Kinematic Replay] Preparing to play {num_parallel} demos in parallel: {target_demos}")

        # Check if kinematic data is available
        if "object_poses" not in data_grp[target_demos[0]]:
            print("ERROR: Dataset does not contain 'object_poses' and 'robot_joint_poses'.")
            print("Please regenerate the dataset using the updated collect/generate scripts!")
            simulation_app.close()
            return

        all_obj_traj = []
        all_joint_traj = []
        all_init_robot_pos = []
        all_init_robot_quat = []
        max_length = 0
        
        for key in target_demos:
            obj_traj = data_grp[key]["object_poses"][:]
            joint_traj = data_grp[key]["robot_joint_poses"][:]
            all_obj_traj.append(obj_traj)
            all_joint_traj.append(joint_traj)
            if len(obj_traj) > max_length:
                max_length = len(obj_traj)
                
            if "init_robot_pos" in data_grp[key]:
                all_init_robot_pos.append(data_grp[key]["init_robot_pos"][:])
                all_init_robot_quat.append(data_grp[key]["init_robot_quat"][:])
            else:
                all_init_robot_pos.append(None)
                all_init_robot_quat.append(None)

    env_cfg = DualArmILEnvCfg()
    env_cfg.sim.device = args_cli.device
    
    # FOR KINEMATIC REPLAY: Disable physics on the object so it doesn"t get pushed by collisions!
    if hasattr(env_cfg.scene.object, "spawn"):
        from isaaclab.sim import RigidBodyPropertiesCfg
        env_cfg.scene.object.spawn.rigid_props = RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
    env_cfg.scene.num_envs = num_parallel
    gym.register(
        id="Isaac-Dual-Arm-IL-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": DualArmILEnvCfg},
        disable_env_checker=True,
    )
    env: ManagerBasedRLEnv = gym.make("Isaac-Dual-Arm-IL-v0", cfg=env_cfg).unwrapped

    env.reset()
    
    print(f"\n--- Starting KINEMATIC parallel replay (Max steps: {max_length}) ---")
    
    obj_state = env.scene["object"].data.default_root_state.clone()
    rob_state = env.scene["robot"].data.default_root_state.clone()
    j_pos = env.scene["robot"].data.default_joint_pos.clone()
    j_vel = env.scene["robot"].data.default_joint_vel.clone() * 0.0
    
    # Run loop
    for step_idx in range(max_length):
        if not simulation_app.is_running():
            break

        for i in range(num_parallel):
            o_traj = all_obj_traj[i]
            j_traj = all_joint_traj[i]
            idx = min(step_idx, len(o_traj) - 1)
            
            obj_state[i, :3] = torch.tensor(o_traj[idx, :3], device=env.device) + env.scene.env_origins[i]
            obj_state[i, 3:7] = torch.tensor(o_traj[idx, 3:7], device=env.device)
            j_pos[i] = torch.tensor(j_traj[idx], device=env.device)
            
            if all_init_robot_pos[i] is not None:
                rob_state[i, :3] = torch.tensor(all_init_robot_pos[i], device=env.device) + env.scene.env_origins[i]
                rob_state[i, 3:7] = torch.tensor(all_init_robot_quat[i], device=env.device)
            
        env.scene["object"].write_root_state_to_sim(obj_state)
        env.scene["robot"].write_root_state_to_sim(rob_state)
        env.scene["robot"].write_joint_state_to_sim(j_pos, j_vel)
        
        # Step physics to render
        env.sim.step()
        
        if args_cli.delay > 0:
            time.sleep(args_cli.delay)

    print("Finished kinematic replay.")
    time.sleep(2.0)
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
