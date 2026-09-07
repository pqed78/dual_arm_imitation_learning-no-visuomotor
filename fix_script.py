import re

with open('scripts/generate_scripted_demos.py', 'r') as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    if "save_episode_to_hdf5(" in line:
        line = line.replace("save_episode_to_hdf5(", "# save_episode_to_hdf5(")
    out_lines.append(line)

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.writelines(out_lines)
