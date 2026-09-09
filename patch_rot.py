import re

with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

# For LEFT_HOVER_TARGET and LEFT_LOWER_TARGET
# left_orient_target = ...
# delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.2, max_rot_step=0.020, directed=True)
text = text.replace(
    'delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.2, max_rot_step=0.020, directed=True)',
    'delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.6, max_rot_step=0.080, directed=True)'
)

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.write(text)

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text2 = f.read()

text2 = text2.replace(
    'delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.2, max_rot_step=0.020, directed=True)',
    'delta_l_rot = compute_desired_grasp_rot(wrist_quat_l, left_orient_target, gain=0.6, max_rot_step=0.080, directed=True)'
)

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text2)
