with open('scripts/generate_scripted_demos2.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "if phase_timer[i] > 60:" in line and i < 550:
        lines[i] = line.replace("60", "25")
    elif "if phase_timer[i] > 75:" in line and i < 550:
        lines[i] = line.replace("75", "40")
    elif "if phase_timer[i] > 90:" in line and i > 550 and i < 650:
        lines[i] = line.replace("90", "35")
    elif "if phase_timer[i] > 120:" in line and i > 550:
        lines[i] = line.replace("120", "50")
    elif "if phase_timer[i] > 30:" in line and i > 600:
        lines[i] = line.replace("30", "15")
    elif "if phase_timer[i] > 25:" in line and i > 650:
        lines[i] = line.replace("25", "10")

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.writelines(lines)
