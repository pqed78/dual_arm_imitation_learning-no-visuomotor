import re

def patch_file(filename):
    with open(filename, 'r') as f:
        text = f.read()

    # 1. Restore task-space relaxation
    old_ik = """                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                

                    
                dq_l = solve_pose_dls_ik("""
                
    new_ik = """                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                
                # Dynamic task-space relaxation: Free the wrist orientation during flight to maximize reach
                if phase == PHASE_LEFT_HOVER_TARGET:
                    j_l_tcp = j_l_tcp.clone()
                    j_l_tcp[:, 3:6, :] = 0.0
                    
                dq_l = solve_pose_dls_ik("""
                
    # Also handle the case where there's no empty lines
    old_ik_2 = """                j_l_wrist = jacobians[:, jacobi_left_hand_idx, :6, :][:, :, jacobi_left_joint_ids]
                j_l_tcp = get_tcp_jacobian(j_l_wrist, wrist_pos_l, tcp_pos_l)
                dq_l = solve_pose_dls_ik("""
                
    if old_ik in text:
        text = text.replace(old_ik, new_ik)
    elif old_ik_2 in text:
        text = text.replace(old_ik_2, new_ik)
        
    # 2. Change max_rot_step=0.015 and gain=0.15 to max_rot_step=0.010 and gain=0.10
    text = text.replace('gain=0.15, max_rot_step=0.015', 'gain=0.10, max_rot_step=0.010')

    with open(filename, 'w') as f:
        f.write(text)

patch_file('scripts/generate_scripted_demos.py')
patch_file('scripts/generate_scripted_demos2.py')
