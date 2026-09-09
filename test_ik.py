import torch
j = torch.randn(1, 6, 7)
j[:, 3:6, :] = 0.0
damping = 0.05
jjt = torch.bmm(j, j.transpose(1, 2))
identity = torch.eye(6).unsqueeze(0)
inv_term = torch.linalg.inv(jjt + damping**2 * identity)
j_pinv = torch.bmm(j.transpose(1, 2), inv_term)
e = torch.randn(1, 6, 1)
dq = torch.bmm(j_pinv, e)
print(dq.shape)
