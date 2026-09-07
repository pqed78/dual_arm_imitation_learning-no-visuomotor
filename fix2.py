with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text = f.read()

text = text.replace("def # save_episode_to_hdf5", "def save_episode_to_hdf5")
text = text.replace("# save_episode_to_hdf5(", "save_episode_to_hdf5(")

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text)
