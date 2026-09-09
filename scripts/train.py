# Copyright (c) 2026, Dual Arm Imitation Learning Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Training Script for Dual Arm Imitation Learning.

Supports:
    - Behavior Cloning (BC)
    - Diffusion Policy
    - Action Chunking with Transformers (ACT)

Usage:
    python scripts/train.py --algo=bc --epochs=100
    python scripts/train.py --algo=diffusion --epochs=150
    python scripts/train.py --algo=act --epochs=150
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.il_dataset import DualArmDataset
from models import MLPBCPolicy, RNNBCPolicy, DiffusionPolicy, ACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train imitation learning policies.")
    parser.add_argument(
        "--algo",
        type=str,
        default="diffusion",
        choices=["bc", "diffusion", "act"],
        help="Algorithm to train: bc, diffusion, or act.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "demos.hdf5"),
        help="Path to HDF5 demonstration dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/{algo}_cfg.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint file to resume training from.")
    return parser.parse_args()


from torch.utils.tensorboard import SummaryWriter

def load_config(algo: str, custom_cfg_path: str | None) -> dict:
    if custom_cfg_path is None:
        cfg_path = os.path.join(PROJECT_ROOT, "configs", f"{algo}_cfg.yaml")
    else:
        cfg_path = custom_cfg_path

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_model(algo: str, cfg: dict, obs_dim: int, act_dim: int) -> torch.nn.Module:
    if algo == "bc":
        model_type = cfg.get("model_type", "mlp")
        if model_type == "rnn":
            return RNNBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=cfg.get("rnn_hidden_dim", 256),
                num_layers=cfg.get("rnn_num_layers", 2),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            return MLPBCPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dims=cfg.get("hidden_dims", [512, 512, 256]),
                dropout=cfg.get("dropout", 0.1),
                activation=cfg.get("activation", "relu"),
            )

    elif algo == "diffusion":
        return DiffusionPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            pred_horizon=cfg.get("pred_horizon", 16),
            obs_horizon=cfg.get("obs_horizon", 2),
            num_train_timesteps=cfg.get("num_train_timesteps", 100),
            beta_schedule=cfg.get("beta_schedule", "squaredcos_cap_v2"),
            down_dims=cfg.get("down_dims", [256, 512, 1024]),
            kernel_size=cfg.get("kernel_size", 5),
            n_groups=cfg.get("n_groups", 8),
        )

    elif algo == "act":
        return ACTPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            chunk_size=cfg.get("chunk_size", 24),
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_encoder_layers=cfg.get("num_encoder_layers", 4),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
            dim_feedforward=cfg.get("dim_feedforward", 1024),
            latent_dim=cfg.get("latent_dim", 32),
            kl_weight=cfg.get("kl_weight", 10.0),
            dropout=cfg.get("dropout", 0.1),
            temporal_ensembling=cfg.get("temporal_ensembling", True),
            temporal_ensemble_coeff=cfg.get("temporal_ensemble_coeff", 0.01),
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.algo, args.config)

    # Overrides
    epochs = args.epochs or cfg.get("epochs", 100)
    batch_size = args.batch_size or cfg.get("batch_size", 64)
    lr = args.lr or cfg.get("learning_rate", 1.0e-4)
    weight_decay = cfg.get("weight_decay", 1.0e-5)

    print("=" * 60)
    print(f" Training Imitation Learning Policy: {args.algo.upper()}")
    print("=" * 60)
    print(f" Dataset    : {args.dataset}")
    print(f" Batch Size : {batch_size}")
    print(f" Epochs     : {epochs}")
    print(f" Device     : {args.device}")
    print(f" LR         : {lr}")
    print("=" * 60)

    # Setup directories
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TensorBoard Writer
    tb_dir = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f" TensorBoard logged to: {tb_dir}")

    # Load Dataset
    pred_h = cfg.get("pred_horizon", 16) if args.algo == "diffusion" else cfg.get("chunk_size", 24)
    obs_h = cfg.get("obs_horizon", 2)

    full_dataset = DualArmDataset(
        dataset_path=args.dataset,
        algo=args.algo,
        pred_horizon=pred_h,
        obs_horizon=obs_h,
    )

    # Save normalization stats for evaluation
    stats_path = os.path.join(save_dir, "stats.pkl")
    full_dataset.save_stats(stats_path)

    # Train / Val Split (90% train, 10% val)
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate Model
    model = build_model(args.algo, cfg, full_dataset.obs_dim, full_dataset.act_dim)
    model.to(args.device)
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[Model] Resuming training from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=args.device, weights_only=True))
        else:
            print(f"[Warning] Checkpoint not found: {args.resume}. Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.algo.upper()} created with {total_params:,} trainable parameters.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    global_step = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            num_train_batches += 1
            global_step += 1
            pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            # Log step-wise training loss
            writer.add_scalar("Loss/Train_Step", loss.item(), global_step)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, num_train_batches)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(args.device) for k, v in batch.items()}
                loss_dict = model.compute_loss(batch)
                val_loss_sum += loss_dict["loss"].item()
                num_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Log epoch-wise metrics to TensorBoard
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "algo": args.algo,
                    "config": cfg,
                    "obs_dim": full_dataset.obs_dim,
                    "act_dim": full_dataset.act_dim,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best model to {best_ckpt_path} (Val Loss: {best_val_loss:.5f})")

        # Periodic checkpoint
        if epoch % cfg.get("save_interval", 20) == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)

    writer.close()
    print("\n[Training Complete]")
    print(f"Best model saved at: {os.path.join(save_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
