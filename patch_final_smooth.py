import re

def patch_file(filename, is_batched=False):
    with open(filename, 'r') as f:
        text = f.read()
        
    # 1. Remove task-space relaxation
    relaxation_block = """                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                if phase == PHASE_LEFT_HOVER_TARGET:
                    j_l_tcp = j_l_tcp.clone()
                    j_l_tcp[:, 3:6, :] = 0.0"""
    text = text.replace(relaxation_block, "")
    
    # 2. Reduce step_size in HOVER_TARGET
    if not is_batched:
        text = text.replace('step_size = min(0.025, dist.item())', 'step_size = min(0.016, dist.item())')
    else:
        text = text.replace('max=0.025', 'max=0.016')
        
    # 3. Reduce max_rot_step and gain in left arm
    text = text.replace('gain=0.25, max_rot_step=0.030', 'gain=0.15, max_rot_step=0.015')

    with open(filename, 'w') as f:
        f.write(text)

patch_file('scripts/generate_scripted_demos.py', is_batched=False)
patch_file('scripts/generate_scripted_demos2.py', is_batched=True)
