import re

def patch_file(filename, is_batched=False):
    with open(filename, 'r') as f:
        text = f.read()

    # The block to replace in both PHASE_RIGHT_LIFT and PHASE_RIGHT_HANDOVER
    old_block = """                # Left arm stays in safe standby posture while right arm completes handover setup
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)
                delta_l_pos = torch.zeros((1, 3), device=args_cli.device)
                delta_l_rot = torch.zeros((1, 3), device=args_cli.device)"""
                
    old_block_lift = """                # Left arm holds standby
                commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)"""

    if not is_batched:
        new_block = """                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l)
                step_size_l = min(0.012, dist_l.item())
                delta_l_pos = (err_l / (dist_l + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)"""
    else:
        new_block = """                # Left arm pre-moves to handover wait position to save time
                err_l = left_wait_pos - tcp_pos_l
                dist_l = torch.norm(err_l, dim=-1)
                step_size_l = torch.clamp(dist_l, max=0.012).unsqueeze(-1)
                delta_l_pos = (err_l / (dist_l.unsqueeze(-1) + 1e-6)) * step_size_l
                left_orient_target = -torch.sign(target_cube_z[:, 1:2] + 1e-6) * target_cube_z
                delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, directed=True)"""
        
    text = text.replace(old_block, new_block)
    
    # In PHASE_RIGHT_LIFT, there is no delta_l_pos initialization in the original code because it relied on the zero initialization at the top of the loop.
    # We must replace the standby block AND set the deltas.
    # Wait, the standby block in PHASE_RIGHT_LIFT is just old_block_lift.
    # Let's replace old_block_lift ONLY in PHASE_RIGHT_LIFT.
    
    lines = text.split('\n')
    in_right_lift = False
    for i, line in enumerate(lines):
        if 'elif phase == PHASE_RIGHT_LIFT:' in line:
            in_right_lift = True
        elif 'elif phase == PHASE_RIGHT_HANDOVER:' in line:
            in_right_lift = False
            
        if in_right_lift and 'commanded_left = commanded_left + torch.clamp(left_standby_joints - commanded_left, min=-0.04, max=0.04)' in line:
            # We found the standby line in PHASE_RIGHT_LIFT
            # We replace it and the preceding comment with new_block
            lines[i-1] = "" # clear comment
            lines[i] = new_block
            
    text = '\n'.join(lines)

    with open(filename, 'w') as f:
        f.write(text)

patch_file('scripts/generate_scripted_demos.py', is_batched=False)
patch_file('scripts/generate_scripted_demos2.py', is_batched=True)
