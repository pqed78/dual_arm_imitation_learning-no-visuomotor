import re
with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text = f.read()

# Fix squeeze on policy_obs
text = text.replace('policy_obs = obs["policy"].squeeze(0).detach().cpu().numpy()', 'policy_obs = obs["policy"].detach().cpu().numpy()')
text = text.replace('action_np = action.squeeze(0).detach().cpu().numpy()', 'action_np = action.detach().cpu().numpy()')

solve_old = """    j = jacobian.squeeze(0)  # (6, 7)
    e = delta_pose.squeeze(0)  # (6,)
    jjt = torch.matmul(j, j.transpose(0, 1))  # (6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype)
    inv_term = torch.inverse(jjt + (damping ** 2) * identity)
    j_pinv = torch.matmul(j.transpose(0, 1), inv_term)  # (7, 6)
    delta_q = torch.matmul(j_pinv, e)  # (7,)

    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
        null_proj = eye_n - torch.matmul(j_pinv, j)  # (7, 7)
        q_err = (q_nominal - q_current).squeeze(0)  # (7,)
        delta_q_null = torch.matmul(null_proj, k_null * q_err)  # (7,)
        delta_q = delta_q + delta_q_null

    return delta_q.unsqueeze(0)"""

solve_new = """    j = jacobian  # (N, 6, 7)
    e = delta_pose.unsqueeze(-1)  # (N, 6, 1)
    jjt = torch.bmm(j, j.transpose(1, 2))  # (N, 6, 6)
    identity = torch.eye(6, device=j.device, dtype=j.dtype).unsqueeze(0).expand(j.shape[0], 6, 6)
    d = damping ** 2 if not isinstance(damping, torch.Tensor) else damping.view(-1,1,1)**2
    inv_term = torch.linalg.inv(jjt + d * identity)
    j_pinv = torch.bmm(j.transpose(1, 2), inv_term)  # (N, 7, 6)
    delta_q = torch.bmm(j_pinv, e).squeeze(-1)  # (N, 7)

    if q_current is not None and q_nominal is not None:
        eye_n = torch.eye(j.shape[2], device=j.device, dtype=j.dtype).unsqueeze(0).expand(j.shape[0], 7, 7)
        null_proj = eye_n - torch.bmm(j_pinv, j)  # (N, 7, 7)
        q_err = (q_nominal - q_current).unsqueeze(-1)  # (N, 7, 1)
        k = k_null if not isinstance(k_null, torch.Tensor) else k_null.view(-1,1,1)
        delta_q_null = torch.bmm(null_proj, k * q_err).squeeze(-1)  # (N, 7)
        delta_q = delta_q + delta_q_null

    return delta_q"""

text = text.replace(solve_old, solve_new)
with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text)
