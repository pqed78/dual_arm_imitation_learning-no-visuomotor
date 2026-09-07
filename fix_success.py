with open('scripts/generate_scripted_demos2.py', 'r') as f:
    text = f.read()

import re

old = """                    obs_tensor = torch.tensor(ep_obs)  # (steps, num_envs, obs_dim)
                    act_tensor = torch.tensor(ep_actions)
                    rew_tensor = torch.tensor(ep_rewards)
                    for env_idx in range(num_envs):
                        save_episode_to_hdf5(
                            args_cli.dataset_file,
                            collected_count,
                            [obs_tensor[:, env_idx].numpy()], # list of (obs_dim,) to match old API, actually saver expects list of arrays
                            [act_tensor[:, env_idx].numpy()],
                            [rew_tensor[:, env_idx].numpy()],
                        )
                        collected_count += 1"""

new = """                    import numpy as np
                    obs_np = np.array(ep_obs)  # (steps, num_envs, obs_dim)
                    act_np = np.array(ep_actions)
                    rew_np = np.array(ep_rewards)
                    for env_idx in range(num_envs):
                        save_episode_to_hdf5(
                            args_cli.dataset_file,
                            collected_count,
                            list(obs_np[:, env_idx, :]),
                            list(act_np[:, env_idx, :]),
                            list(rew_np[:, env_idx]),
                        )
                        collected_count += 1"""

text = text.replace(old, new)

with open('scripts/generate_scripted_demos2.py', 'w') as f:
    f.write(text)
