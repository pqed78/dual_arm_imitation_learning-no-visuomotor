with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

lines = text.split('\n')
in_lower = False
for i, line in enumerate(lines):
    if 'elif phase == PHASE_LEFT_LOWER_TARGET:' in line:
        in_lower = True
    elif 'elif phase == PHASE_LEFT_RELEASE:' in line:
        in_lower = False
    
    if in_lower and 'step_size = min(0.025, dist.item())' in line:
        lines[i] = line.replace('min(0.025, dist.item())', 'min(0.008, dist.item())')

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.write('\n'.join(lines))

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text2 = f.read()

lines2 = text2.split('\n')
in_lower = False
for i, line in enumerate(lines2):
    if 'elif phase == PHASE_LEFT_LOWER_TARGET:' in line:
        in_lower = True
    elif 'elif phase == PHASE_LEFT_RELEASE:' in line:
        in_lower = False
    
    if in_lower and 'step_size = torch.clamp(dist, max=0.025)' in line:
        lines2[i] = line.replace('max=0.025', 'max=0.008')

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write('\n'.join(lines2))
