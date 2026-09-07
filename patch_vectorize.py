import re

with open('scripts/generate_scripted_demos.py', 'r') as f:
    text = f.read()

# 1. Update EnvCfg instantiation to support num_envs
cfg_pattern = "cfg = DualArmILEnvCfg()"
cfg_replace = """cfg = DualArmILEnvCfg()
    if hasattr(args_cli, "num_envs") and args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    num_envs = cfg.scene.num_envs"""
text = text.replace(cfg_pattern, cfg_replace)

# We will just write a fully parallel wrapper around the `step` loop.
# Actually, since modifying the 500-line logic via regex is very brittle,
# let's just make it a batched loop for `ep_obs` and `is_success`,
# but wait! The easiest way is to modify `generate_scripted_demos.py` using regex on `phase` -> `phase[env_idx]`.

# Let's just output a message indicating it's too complex for a one-liner regex.
print("Regex too complex. Doing it manually.")
