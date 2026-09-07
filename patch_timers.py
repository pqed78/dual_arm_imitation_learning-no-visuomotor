import re

with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

text = text.replace('if phase_timer > 60:', 'if phase_timer > 25:', 1)  # RIGHT_GRASP
text = text.replace('if phase_timer > 75:', 'if phase_timer > 40:', 1)  # RIGHT_GRASP retry
text = text.replace('if phase_timer > 90:', 'if phase_timer > 35:', 1)  # LEFT_GRASP
text = text.replace('if phase_timer > 120:', 'if phase_timer > 50:', 1) # LEFT_GRASP retry
text = text.replace('if phase_timer > 30:', 'if phase_timer > 15:', 1)  # RIGHT_RELEASE
text = text.replace('if phase_timer > 30:', 'if phase_timer > 15:', 1)  # RIGHT_RETREAT
text = text.replace('if phase_timer > 25:', 'if phase_timer > 10:', 1)  # LEFT_RELEASE

with open('scripts/generate_scripted_demos.py', 'w') as f:
    f.write(text)
