import re

with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

# 1. Add num_envs handling
cfg_pattern = "cfg = DualArmILEnvCfg()"
cfg_replace = """cfg = DualArmILEnvCfg()
    if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    num_envs = cfg.scene.num_envs"""
text = text.replace(cfg_pattern, cfg_replace)

# 2. Modify state machine init
init_pattern = """        phase = PHASE_INIT
        phase_timer = 0
        latched_align_sign = None
        latched_cube_z = None
        target_cube_z = None
        target_cube_z_for_wrist = None"""
init_replace = """        phase = [PHASE_INIT] * num_envs
        phase_timer = [0] * num_envs
        latched_align_sign = [None] * num_envs
        latched_cube_z = [None] * num_envs
        target_cube_z = [None] * num_envs
        target_cube_z_for_wrist = [None] * num_envs
        pick_target = [None] * num_envs
        place_target = [None] * num_envs
        is_active = [True] * num_envs
        is_success = [False] * num_envs
        
        commanded_left = left_standby_joints.expand(num_envs, 7).clone()
        commanded_right = right_standby_joints.expand(num_envs, 7).clone()
        
        left_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)
        right_gripper_cmd = torch.ones((num_envs, 1), device=args_cli.device)"""
text = text.replace(init_pattern, init_replace)

text = text.replace("        commanded_left = left_standby_joints.clone()\n        commanded_right = right_standby_joints.clone()\n", "")
text = text.replace("        left_gripper_cmd = 1.0   # Open (+1.0)\n        right_gripper_cmd = 1.0  # Open (+1.0)\n", "")

# 3. Inside step loop, before logic
step_pattern = """            # Object and Target Poses
            obj_pos = obj.data.root_pos_w.clone()        # (1, 3)
            obj_quat = obj.data.root_quat_w.clone()      # (1, 4)
            target_pos = target.data.root_pos_w.clone()  # (1, 3)"""
step_replace = """            delta_r_pos_all = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_r_rot_all = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_l_pos_all = torch.zeros((num_envs, 3), device=args_cli.device)
            delta_l_rot_all = torch.zeros((num_envs, 3), device=args_cli.device)

            obj_pos_all = obj.data.root_pos_w.clone()
            obj_quat_all = obj.data.root_quat_w.clone()
            target_pos_all = target.data.root_pos_w.clone()
            wrist_pos_r_all = robot.data.body_pos_w[:, right_hand_idx]
            wrist_quat_r_all = robot.data.body_quat_w[:, right_hand_idx]
            tcp_pos_r_all, _ = compute_tcp(wrist_pos_r_all, wrist_quat_r_all)
            wrist_pos_l_all = robot.data.body_pos_w[:, left_hand_idx]
            wrist_quat_l_all = robot.data.body_quat_w[:, left_hand_idx]
            tcp_pos_l_all, _ = compute_tcp(wrist_pos_l_all, wrist_quat_l_all)
            right_gripper_width_all = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1)
            left_gripper_width_all = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1)

            for i in range(num_envs):
                if not is_active[i]: continue

                phase_timer[i] += 1
                obj_pos = obj_pos_all[i:i+1]
                obj_quat = obj_quat_all[i:i+1]
                target_pos = target_pos_all[i:i+1]
                wrist_pos_r = wrist_pos_r_all[i:i+1]
                wrist_quat_r = wrist_quat_r_all[i:i+1]
                tcp_pos_r = tcp_pos_r_all[i:i+1]
                wrist_pos_l = wrist_pos_l_all[i:i+1]
                wrist_quat_l = wrist_quat_l_all[i:i+1]
                tcp_pos_l = tcp_pos_l_all[i:i+1]
                right_gripper_width = right_gripper_width_all[i].item()
                left_gripper_width = left_gripper_width_all[i].item()"""
text = text.replace(step_pattern, step_replace)

# 4. Remove old variable extractions that are now batched
text = text.replace("            phase_timer += 1\n", "")
text = text.replace("            wrist_pos_r = robot.data.body_pos_w[:, right_hand_idx]\n            wrist_quat_r = robot.data.body_quat_w[:, right_hand_idx]\n            tcp_pos_r, z_dir_r = compute_tcp(wrist_pos_r, wrist_quat_r)\n", "")
text = text.replace("            wrist_pos_l = robot.data.body_pos_w[:, left_hand_idx]\n            wrist_quat_l = robot.data.body_quat_w[:, left_hand_idx]\n            tcp_pos_l, z_dir_l = compute_tcp(wrist_pos_l, wrist_quat_l)\n", "")
text = text.replace("            right_gripper_width = torch.sum(robot.data.joint_pos[:, right_gripper_joint_ids], dim=-1).item()\n            left_gripper_width = torch.sum(robot.data.joint_pos[:, left_gripper_joint_ids], dim=-1).item()\n", "")

# 5. Indent the main logic block
import io
lines = io.StringIO(text).readlines()
new_lines = []
in_logic = False
for line in lines:
    if "Fail-fast: If baton drops" in line:
        in_logic = True
    if "Real-time 3D state" in line:
        in_logic = False
        new_lines.append("                delta_r_pos_all[i] = delta_r_pos[0]\n")
        new_lines.append("                delta_r_rot_all[i] = delta_r_rot[0]\n")
        new_lines.append("                delta_l_pos_all[i] = delta_l_pos[0]\n")
        new_lines.append("                delta_l_rot_all[i] = delta_l_rot[0]\n")
        
    if in_logic:
        # replace phase with phase[i] and timers
        line = line.replace("phase = ", "phase[i] = ")
        line = line.replace("phase_timer = ", "phase_timer[i] = ")
        line = line.replace("phase == ", "phase[i] == ")
        line = line.replace("phase < ", "phase[i] < ")
        line = line.replace("phase <=", "phase[i] <=")
        line = line.replace("<= phase", "<= phase[i]")
        line = line.replace("phase_timer >", "phase_timer[i] >")
        line = line.replace("PHASE_NAMES[phase]", "PHASE_NAMES[phase[i]]")
        
        line = line.replace("latched_align_sign is None", "latched_align_sign[i] is None")
        line = line.replace("latched_align_sign is not None", "latched_align_sign[i] is not None")
        line = line.replace("latched_align_sign =", "latched_align_sign[i] =")
        line = line.replace("latched_cube_z =", "latched_cube_z[i] =")
        line = line.replace("target_cube_z =", "target_cube_z[i] =")
        line = line.replace("target_cube_z_for_wrist =", "target_cube_z_for_wrist[i] =")
        line = line.replace("pick_target =", "pick_target[i] =")
        line = line.replace("place_target =", "place_target[i] =")
        
        # Replace usages
        line = line.replace("latched_align_sign", "latched_align_sign[i]")
        line = line.replace("latched_cube_z", "latched_cube_z[i]")
        line = line.replace("target_cube_z_for_wrist", "target_cube_z_for_wrist[i]")
        line = line.replace("target_cube_z", "target_cube_z[i]")
        line = line.replace("pick_target", "pick_target[i]")
        line = line.replace("place_target", "place_target[i]")
        
        line = line.replace("commanded_left", "commanded_left[i:i+1]")
        line = line.replace("commanded_right", "commanded_right[i:i+1]")
        line = line.replace("left_gripper_cmd =", "left_gripper_cmd[i, 0] =")
        line = line.replace("right_gripper_cmd =", "right_gripper_cmd[i, 0] =")
        
        line = line.replace("break", "is_active[i] = False; continue")
        new_lines.append("    " + line)
    else:
        new_lines.append(line)
        
with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.writelines(new_lines)
