with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()
text = text.replace('step_size = min(0.025, dist.item())', 'step_size = min(0.018, dist.item())')
with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.write(text)

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text2 = f.read()
# First fix the fact it wasn't patched: Replace 0.012 with 0.018 in PHASE_LEFT_HOVER_TARGET and PHASE_LEFT_LOWER_TARGET
# Let's just find lines 673 and 692 (or around there) and patch them.
lines = text2.split('\n')
in_left = False
for i, line in enumerate(lines):
    if 'elif phase == PHASE_LEFT_HOVER_TARGET:' in line:
        in_left = True
    elif 'elif phase == PHASE_LEFT_RELEASE:' in line:
        in_left = False
    
    if in_left and 'step_size = torch.clamp(dist, max=0.012)' in line:
        lines[i] = line.replace('max=0.012', 'max=0.018')

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write('\n'.join(lines))
