with open('scripts/generate_scripted_demos.py', 'r') as f:
    lines = f.readlines()

in_left_hover = False
for i, line in enumerate(lines):
    if "elif phase == PHASE_LEFT_HOVER_TARGET:" in line:
        in_left_hover = True
    elif "elif phase == PHASE_LEFT_RELEASE:" in line:
        in_left_hover = False
        
    if in_left_hover and "step_size = min(0.012, dist.item())" in line:
        lines[i] = line.replace("0.012", "0.025")

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.writelines(lines)

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    lines2 = f.readlines()

in_left_hover = False
for i, line in enumerate(lines2):
    if "elif phase[i] == PHASE_LEFT_HOVER_TARGET:" in line:
        in_left_hover = True
    elif "elif phase[i] == PHASE_LEFT_RELEASE:" in line:
        in_left_hover = False
        
    if in_left_hover and "step_size = torch.clamp(dist, max=0.012)" in line:
        lines2[i] = line.replace("max=0.012", "max=0.025")

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.writelines(lines2)
