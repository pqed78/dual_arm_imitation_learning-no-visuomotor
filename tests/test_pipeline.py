# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test to verify tensor dimensions and forward/backward passes for all models."""

import os
import sys
import tempfile
import h5py
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def test_models():
    print("[Test] Testing Models...")
    batch_size = 4
    obs_dim = 45   # approximate policy observation dimension
    act_dim = 16   # 14 arm joints + 2 grippers
    pred_h = 16
    obs_h = 2
    chunk_size = 24

    # 1. MLP BC
    mlp_bc = MLPBCPolicy(obs_dim=obs_dim, act_dim=act_dim)
    dummy_obs = torch.randn(batch_size, obs_dim)
    dummy_act = torch.randn(batch_size, act_dim)
    loss = mlp_bc.compute_loss({"obs": dummy_obs, "action": dummy_act})
    loss["loss"].backward()
    print("  [✓] MLP BC forward/backward passed. Loss:", loss["loss"].item())

    # 2. Diffusion Policy
    diffusion = DiffusionPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        pred_horizon=pred_h,
        obs_horizon=obs_h,
        num_train_timesteps=20,
        down_dims=[64, 128],
    )
    dummy_obs_seq = torch.randn(batch_size, obs_h, obs_dim)
    dummy_act_seq = torch.randn(batch_size, pred_h, act_dim)
    diff_loss = diffusion.compute_loss({"obs": dummy_obs_seq, "action": dummy_act_seq})
    diff_loss["loss"].backward()
    print("  [✓] Diffusion Policy forward/backward passed. Loss:", diff_loss["loss"].item())

    # Test sampling
    pred_actions = diffusion.predict_action(dummy_obs_seq[:1], num_inference_steps=5)
    assert pred_actions.shape == (1, pred_h, act_dim), f"Expected (1, {pred_h}, {act_dim}), got {pred_actions.shape}"
    print("  [✓] Diffusion Policy sampling passed. Shape:", pred_actions.shape)

    # 3. ACT Policy
    act_policy = ACTPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        chunk_size=chunk_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        latent_dim=16,
    )
    dummy_chunk_act = torch.randn(batch_size, chunk_size, act_dim)
    act_loss = act_policy.compute_loss({"obs": dummy_obs, "action": dummy_chunk_act})
    act_loss["loss"].backward()
    print("  [✓] ACT Policy forward/backward passed. Loss:", act_loss["loss"].item())

    # Test closed-loop step prediction with ensembling
    act_step = act_policy.predict_action_step(dummy_obs[0], current_step=0)
    assert act_step.shape == (act_dim,), f"Expected ({act_dim},), got {act_step.shape}"
    print("  [✓] ACT Policy action step prediction passed. Shape:", act_step.shape)


def test_dataset():
    print("\n[Test] Testing HDF5 Dataset...")
    obs_dim = 45
    act_dim = 16

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create a mock HDF5 dataset with 2 demos
        with h5py.File(tmp_path, "w") as f:
            data_grp = f.create_group("data")
            for d in range(2):
                d_grp = data_grp.create_group(f"demo_{d}")
                d_grp.create_dataset("obs", data=np.random.randn(50, obs_dim).astype(np.float32))
                d_grp.create_dataset("actions", data=np.random.randn(50, act_dim).astype(np.float32))

        # Test loading for BC
        ds_bc = DualArmDataset(tmp_path, algo="bc")
        item_bc = ds_bc[0]
        assert item_bc["obs"].shape == (obs_dim,)
        assert item_bc["action"].shape == (act_dim,)
        print("  [✓] Dataset (BC mode) passed. Sample:", item_bc["obs"].shape, item_bc["action"].shape)

        # Test loading for Diffusion
        ds_diff = DualArmDataset(tmp_path, algo="diffusion", pred_horizon=16, obs_horizon=2)
        item_diff = ds_diff[0]
        assert item_diff["obs"].shape == (2, obs_dim)
        assert item_diff["action"].shape == (16, act_dim)
        print("  [✓] Dataset (Diffusion mode) passed. Sample:", item_diff["obs"].shape, item_diff["action"].shape)

        # Test loading for ACT
        ds_act = DualArmDataset(tmp_path, algo="act", pred_horizon=24)
        item_act = ds_act[0]
        assert item_act["obs"].shape == (obs_dim,)
        assert item_act["action"].shape == (24, act_dim)
        print("  [✓] Dataset (ACT mode) passed. Sample:", item_act["obs"].shape, item_act["action"].shape)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    test_models()
    test_dataset()
    print("\n=== All unit tests passed successfully! ===")
