with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

text = text.replace('step_size = min(0.018, dist.item())', 'step_size = min(0.025, dist.item())')
text = text.replace('gain=0.6, max_rot_step=0.080', 'gain=0.25, max_rot_step=0.030')

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.write(text)

with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text2 = f.read()

text2 = text2.replace('max=0.018', 'max=0.025')
text2 = text2.replace('gain=0.6, max_rot_step=0.080', 'gain=0.25, max_rot_step=0.030')

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text2)
