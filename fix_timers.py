with open('scripts/generate_scripted_demos.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "if phase_timer > 10:" in line and i < 550:
        lines[i] = line.replace("10", "25")
    elif "if phase_timer > 25:" in line and i > 650:
        lines[i] = line.replace("25", "10")

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.writelines(lines)
