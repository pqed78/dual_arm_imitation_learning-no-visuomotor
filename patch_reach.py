import re

def patch_file(filename):
    with open(filename, 'r') as f:
        text = f.read()

    # Find the block where j_l_tcp is computed before solve_pose_dls_ik
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
    
    text = text.replace(old_ik, new_ik)

    with open(filename, 'w') as f:
        f.write(text)

patch_file('scripts/generate_scripted_demos.py')
patch_file('scripts/generate_scripted_demos2.py')
