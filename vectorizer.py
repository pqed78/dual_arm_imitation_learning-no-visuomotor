import re

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text = f.read()

# 1. Update EnvCfg instantiation to support num_envs
cfg_pattern = "cfg = DualArmILEnvCfg()"
cfg_replace = """cfg = DualArmILEnvCfg()
    if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    num_envs = cfg.scene.num_envs"""
text = text.replace(cfg_pattern, cfg_replace)

# 2. Fix dist item for step_size
text = text.replace("step_size = min(0.012, dist.item())", "step_size = torch.clamp(dist, max=0.012).unsqueeze(-1)")
text = text.replace("step_size = min(0.012, torch.norm(err).item())", "step_size = torch.clamp(torch.norm(err, dim=-1), max=0.012).unsqueeze(-1)")
text = text.replace("step_size = min(0.008, torch.norm(err).item())", "step_size = torch.clamp(torch.norm(err, dim=-1), max=0.012).unsqueeze(-1)")

# 3. Fix division by dist
text = text.replace("(dist + 1e-6)", "(dist.unsqueeze(-1) + 1e-6)")
text = text.replace("(torch.norm(err) + 1e-6)", "(torch.norm(err, dim=-1).unsqueeze(-1) + 1e-6)")

# 4. Fix dist_xy / dist_z which are (N,)
text = text.replace("dist_xy = torch.norm(err[:, :2])", "dist_xy = torch.norm(err[:, :2], dim=-1)")
text = text.replace("dist_z = torch.abs(err[:, 2])", "dist_z = torch.abs(err[:, 2])")
text = text.replace("dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2])", "dist_xy = torch.norm(left_grasp_tgt[:, :2] - tcp_pos_l[:, :2], dim=-1)")

# 5. Fix conditions with .max().item()
text = text.replace("if dist < 0.025", "if dist.max().item() < 0.025")
text = text.replace("if dist < 0.03", "if dist.max().item() < 0.03")
text = text.replace("if (dist < 0.015 or dist_z < 0.006)", "if (dist.max().item() < 0.015 or dist_z.max().item() < 0.006)")
text = text.replace("if (dist < 0.03 and rot_aligned.item())", "if (dist.max().item() < 0.03 and rot_aligned.all().item())")

# 6. Fix `reached` conditions
text = text.replace("reached = (dist_xy < 0.012 and dist_z < 0.005)", "reached = (dist_xy < 0.012) & (dist_z < 0.005)")
text = text.replace("reached = (dist_xy < 0.010 and dist_z < 0.005)", "reached = (dist_xy < 0.010) & (dist_z < 0.005)")
text = text.replace("reached = (dist_xy < 0.025 and dist_z < 0.030)", "reached = (dist_xy < 0.025) & (dist_z < 0.030)")
text = text.replace("reached_l = (dist_xy < 0.025 and dist_z < 0.015)", "reached_l = (dist_xy < 0.025) & (dist_z < 0.015)")
text = text.replace("if reached or phase_timer > 160:", "if reached.all().item() or phase_timer > 160:")
text = text.replace("if reached_l or phase_timer > 160:", "if reached_l.all().item() or phase_timer > 160:")
text = text.replace("if reached:", "if reached.all().item():")
text = text.replace("if reached_l:", "if reached_l.all().item():")

# 7. Gripper width checks
text = text.replace("if right_gripper_width > 0.025:", "if (right_gripper_width > 0.025).all().item():")
text = text.replace("right_gripper_width*1000:.1f", "right_gripper_width.mean().item()*1000:.1f")
text = text.replace("right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1).item()", "right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1)")
text = text.replace("left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1).item()", "left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1)")

# 8. Drop detection
text = text.replace("if 5 <= phase <= 11 and obj_pos[0, 2].item() < 0.05:", "if 5 <= phase <= 11 and obj_pos[:, 2].min().item() < 0.05:")

# 9. Success check
success_old = """            if phase == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos[0, :2] - target_pos[0, :2]).item()
                if dist_to_target < 0.15 and obj_pos[0, 2].item() < 0.06:
                    # save_episode_to_hdf5(
                    #     args_cli.dataset_file,
                    #     collected_count,
                    #     ep_obs,
                    #     ep_actions,
                    #     ep_rewards,
                    # )
                    collected_count += 1
                    print(f"  [✓] Successfully collected Demo #{collected_count-1} in {step} steps! (dist={dist_to_target:.3f}m)")
                    break
                else:
                    print(f"  [X] Object not on target (dist={dist_to_target:.3f}m, z={obj_pos[0, 2]:.3f}m). Discarding...")
                    break"""
                    
success_new = """            if phase == PHASE_SUCCESS:
                dist_to_target = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
                if dist_to_target.max().item() < 0.15 and obj_pos[:, 2].max().item() < 0.06:
                    obs_tensor = torch.tensor(ep_obs)  # (steps, num_envs, obs_dim)
                    act_tensor = torch.tensor(ep_actions)
                    rew_tensor = torch.tensor(ep_rewards)
                    for env_idx in range(num_envs):
                        save_episode_to_hdf5(
                            args_cli.dataset_file,
                            collected_count,
                            [obs_tensor[:, env_idx].numpy()], # list of (obs_dim,) to match old API, actually saver expects list of arrays
                            [act_tensor[:, env_idx].numpy()],
                            [rew_tensor[:, env_idx].numpy()],
                        )
                        collected_count += 1
                    print(f"  [✓] Successfully collected {num_envs} Demos in {step} steps! Total: {collected_count}")
                    break
                else:
                    print(f"  [X] Objects not on target! Max dist={dist_to_target.max().item():.3f}m. Discarding batch...")
                    break"""
text = text.replace(success_old, success_new)

# 10. `norm` without dim=-1
text = text.replace("dist = torch.norm(err)", "dist = torch.norm(err, dim=-1)")

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text)
