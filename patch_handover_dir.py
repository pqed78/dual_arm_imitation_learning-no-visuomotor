import re

def patch_file(filename):
    with open(filename, 'r') as f:
        text = f.read()

    # 1. Update PHASE_RIGHT_HOVER
    text = text.replace(
        'delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_cube_z)',
        'delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, directed=True)'
    )

    # 2. Update PHASE_RIGHT_DESCEND
    text = text.replace(
        'delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_cube_z, gain=0.2, max_rot_step=0.020)',
        'delta_r_rot = compute_desired_grasp_rot(wrist_quat_r, latched_align_sign * latched_cube_z, gain=0.2, max_rot_step=0.020, directed=True)'
    )
    
    # 3. Update PHASE_RIGHT_GRASP
    # In generate_scripted_demos2.py, it's:
    # latched_wrist_sign[i] = torch.sign(torch.sum(x_curr * latched_cube_z[i], dim=-1)).clone()
    # target_cube_z_for_wrist[i] = latched_wrist_sign[i] * target_cube_z[i]
    # In generate_scripted_demos.py, it's:
    # latched_wrist_sign = torch.sign(torch.sum(x_curr * latched_cube_z, dim=-1)).clone()
    # target_cube_z_for_wrist = latched_wrist_sign * target_cube_z
    
    # For script 1
    text = text.replace(
        'latched_wrist_sign = torch.sign(torch.sum(x_curr * latched_cube_z, dim=-1)).clone()',
        'latched_wrist_sign = latched_align_sign.clone()'
    )
    
    # For script 2
    text = text.replace(
        'latched_wrist_sign[i] = torch.sign(torch.sum(x_curr * latched_cube_z[i], dim=-1)).clone()',
        'latched_wrist_sign[i] = latched_align_sign[i].clone()'
    )

    with open(filename, 'w') as f:
        f.write(text)

patch_file('scripts/generate_scripted_demos.py')
patch_file('scripts/generate_scripted_demos2.py')
