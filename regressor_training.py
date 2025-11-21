"""
train_biomime_regressor.py

Pure regression model from BioMime parameters -> MUAPs.

- Uses StaticMuapDataset (pre-generated MUAPs).
- Model: BioMimeRegressor
    c (6) -> MUAP (10,32,96) via a low-rank time basis.
- Loss: MSE on standardized MUAPs.
- Metrics: val nRMSE in original scale.

Run, e.g.:

python train_biomime_regressor.py \
    --data_path data/biomime_static.pt \
    --save_path biomime_regressor_ckpt.pt \
    --epochs 150 \
    --batch_size 64
"""

import math
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# -------------------------------------------------------------------
# Adjust this import to where your dataset lives
# from your_module import StaticMuapDataset
from biomime_data_generator import StaticMuapDataset  # <- CHANGE THIS
# -------------------------------------------------------------------


# ------------------------------
# Metrics
# ------------------------------

def batch_nrmse(x_pred: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute batch-wise normalized RMSE:
      nRMSE = RMSE(x_pred - x_true) / RMS(x_true).
    Returns scalar tensor (averaged over batch).
    """
    diff = x_pred - x_true
    mse = diff.view(diff.size(0), -1).pow(2).mean(dim=1)
    rmse = torch.sqrt(mse + eps)

    true_power = x_true.view(x_true.size(0), -1).pow(2).mean(dim=1)
    denom = torch.sqrt(true_power + eps)

    nrmse = (rmse / denom).mean()
    return nrmse


# ------------------------------
# MUAP normalization: per electrode (C,H)
# ------------------------------

@torch.no_grad()
def compute_muap_normalization_stats(
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-electrode mean/std for MUAPs over the given loader.

    x has shape (B, C, H, W) = (B, 10, 32, 96).
    We compute mean/std over dataset and time dim -> (C, H).

    Returns:
        muap_mean: (1, C, H, 1)
        muap_std:  (1, C, H, 1)
    """
    sum_ = None
    sum_sq = None
    count = 0

    for batch in loader:
        x = batch["muap"].to(device)  # (B, C, H, W)
        B, C, H, W = x.shape

        if sum_ is None:
            sum_ = torch.zeros(C, H, device=device, dtype=torch.float64)
            sum_sq = torch.zeros(C, H, device=device, dtype=torch.float64)

        # Sum over batch and time
        sum_ += x.sum(dim=(0, 3))            # (C, H)
        sum_sq += (x ** 2).sum(dim=(0, 3))   # (C, H)
        count += B * W                       # total time samples per electrode across batch

    mean = (sum_ / count).to(torch.float32)  # (C, H)
    var = (sum_sq / count - sum_ ** 2 / (count ** 2)).to(torch.float32)
    std = torch.sqrt(var + 1e-8)

    muap_mean = mean.view(1, C, H, 1)  # broadcastable
    muap_std = std.view(1, C, H, 1)
    return muap_mean, muap_std


# ------------------------------
# Pure regression model: c -> x
# ------------------------------

class BioMimeRegressor(nn.Module):
    """
    Deterministic regressor:

        c ∈ R^6  →  x_std_pred ∈ R^{C,H,W}

    Uses a learned low-rank temporal basis:

        x[b, c, h, t] = sum_k coeff[b, k, c, h] * basis[k, t]

    - c_dim: number of condition parameters (6)
    - channels: C = 10
    - height:   H = 32
    - time_length: W = 96
    - n_basis: number of temporal basis functions (e.g. 16 or 32)
    """

    def __init__(
        self,
        c_dim: int = 6,
        channels: int = 10,
        height: int = 32,
        time_length: int = 96,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        n_basis: int = 16,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.channels = channels
        self.height = height
        self.time_length = time_length
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_basis = n_basis

        # Embed c into a latent code
        self.embed = nn.Sequential(
            nn.Linear(c_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
        )

        self.temporal_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)),  # conv over time only
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)),
        )

        # Predict coefficients for each (basis, channel, height)
        # Shape is (B, n_basis * C * H)
        self.coeff_head = nn.Linear(hidden_dim, n_basis * channels * height)

        # Learned temporal basis: (n_basis, time_length)
        self.time_basis = nn.Parameter(
            torch.randn(n_basis, time_length) * 0.1
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: (B, 6) normalized conditions.

        Returns:
            x_std_pred: (B, C, H, W) standardized MUAP prediction.
        """
        B = c.size(0)
        C, H, W = self.channels, self.height, self.time_length

        h = self.embed(c)                              # (B, hidden_dim)
        coeff_flat = self.coeff_head(h)                # (B, n_basis * C * H)
        coeff = coeff_flat.view(B, self.n_basis, C, H) # (B, K, C, H)

        # time_basis: (K, W)
        # x[b,c,h,t] = sum_k coeff[b, k, c, h] * time_basis[k, t]
        # Implement with einsum: (B,K,C,H) x (K,W) -> (B,C,H,W)
        x_std_pred = torch.einsum("bkch,kw->bchw", coeff, self.time_basis)

        return x_std_pred



import torch
import torch.nn as nn
import torch.nn.functional as F

class BioMimeCNNRegressor(nn.Module):
    """
    Deterministic regressor:

        c ∈ R^6  →  x_std_pred ∈ R^{C,H,W}

    - Uses a coordinate + condition embedding as input to a small conv2d stack.
    - No low-rank time basis: can represent arbitrary temporal shapes.
    - H is the "spatial" dimension (e.g. 32), W is time (e.g. 96).
    - C is number of output channels (10 electrodes).
    """

    def __init__(
        self,
        c_dim: int = 6,
        channels: int = 10,
        height: int = 32,
        time_length: int = 96,
        emb_dim: int = 128,
        hidden_channels: int = 64,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.channels = channels
        self.height = height
        self.time_length = time_length
        self.emb_dim = emb_dim
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        # --- Condition embedding ---
        self.embed = nn.Sequential(
            nn.Linear(c_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

        # --- Coordinate channels (2: space index, time index) ---
        # shape (2, H, W), values in [-1,1]
        h = torch.linspace(-1.0, 1.0, steps=height).view(1, height, 1).expand(1, height, time_length)
        t = torch.linspace(-1.0, 1.0, steps=time_length).view(1, 1, time_length).expand(1, height, time_length)
        coord = torch.cat([h, t], dim=0)              # (2, H, W)
        self.register_buffer("coord", coord)          # not a parameter

        in_channels = emb_dim + 2  # cond embedding + 2 coord channels

        # --- Conv stack ---
        layers = []
        # First layer: (emb_dim+2) → hidden_channels
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.SiLU())

        # Additional blocks: hidden_channels → hidden_channels
        for _ in range(n_blocks - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.SiLU())

        self.conv_layers = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: (B, 6) normalized conditions.

        Returns:
            x_std_pred: (B, C, H, W) standardized MUAP prediction.
        """
        B = c.size(0)
        H, W = self.height, self.time_length

        # Condition embedding
        emb = self.embed(c)                          # (B, emb_dim)
        emb_map = emb.unsqueeze(-1).unsqueeze(-1)    # (B, emb_dim, 1, 1)
        emb_map = emb_map.expand(-1, -1, H, W)       # (B, emb_dim, H, W)

        # Coordinate map (2, H, W) -> (B, 2, H, W)
        coord = self.coord.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

        # Concatenate [coords, cond_emb] along channel axis
        x = torch.cat([coord, emb_map], dim=1)       # (B, emb_dim+2, H, W)

        # Conv stack
        x = self.conv_layers(x)                      # (B, hidden_channels, H, W)

        # Final projection to C output channels
        x = self.out_conv(x)                         # (B, C, H, W)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Simple residual block:
      x -> Conv2d -> SiLU -> Conv2d -> +x
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return x + out


class BioMimeResNetCNNRegressor(nn.Module):
    """
    Deterministic regressor:

        c ∈ R^6  →  x_std_pred ∈ R^{C,H,W}

    - Condition c is embedded with an MLP.
    - We concatenate cond embedding with 2 coordinate channels (row, time).
    - Then we pass through:
        entry Conv2d  → n residual blocks → 1x1 Conv2d to C channels.

    Args:
        c_dim:           dimension of condition vector (6)
        channels:        output channels (10)
        height:          spatial dimension (32)
        time_length:     time dimension (96)
        emb_dim:         condition embedding size
        hidden_channels: feature maps in conv stack
        n_blocks:        number of ResBlocks
    """

    def __init__(
        self,
        c_dim: int = 6,
        channels: int = 10,
        height: int = 32,
        time_length: int = 96,
        emb_dim: int = 128,
        hidden_channels: int = 128,
        n_blocks: int = 6,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.channels = channels
        self.height = height
        self.time_length = time_length
        self.emb_dim = emb_dim
        self.hidden_channels = hidden_channels
        self.n_blocks = n_blocks

        # ----- condition embedding -----
        self.embed = nn.Sequential(
            nn.Linear(c_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

        # ----- coordinate channels (row, time) in [-1, 1] -----
        H, W = height, time_length
        h = torch.linspace(-1.0, 1.0, steps=H).view(1, H, 1).expand(1, H, W)
        t = torch.linspace(-1.0, 1.0, steps=W).view(1, 1, W).expand(1, H, W)
        coord = torch.cat([h, t], dim=0)  # (2, H, W)
        self.register_buffer("coord", coord)

        in_channels = emb_dim + 2  # 2 coords + cond embedding

        # ----- conv stack: entry + residual blocks -----
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(hidden_channels))
        self.blocks = nn.Sequential(*blocks)

        # final projection to MUAP channels
        self.out_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: (B, 6) normalized conditions in [0.5, 1].

        Returns:
            x_std_pred: (B, C, H, W) standardized MUAP prediction.
        """
        B = c.size(0)
        H, W = self.height, self.time_length

        # condition embedding -> broadcast over (H, W)
        emb = self.embed(c)                          # (B, emb_dim)
        emb_map = emb.unsqueeze(-1).unsqueeze(-1)    # (B, emb_dim, 1, 1)
        emb_map = emb_map.expand(-1, -1, H, W)       # (B, emb_dim, H, W)

        # coordinate channels -> (B, 2, H, W)
        coord = self.coord.unsqueeze(0).expand(B, -1, -1, -1)

        x = torch.cat([coord, emb_map], dim=1)       # (B, emb_dim+2, H, W)

        x = self.entry(x)                            # (B, hidden_channels, H, W)
        x = self.blocks(x)                           # residual stack
        x = self.out_conv(x)                         # (B, C, H, W)

        return x


# ------------------------------
# Training & evaluation
# ------------------------------

def train_one_epoch(
    model: BioMimeRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    muap_mean: torch.Tensor,
    muap_std: torch.Tensor,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    muap_mean = muap_mean.to(device)
    muap_std = muap_std.to(device)

    for batch in loader:
        x = batch["muap"].to(device)   # (B, C, H, W)
        c = batch["cond"].to(device)   # (B, 6)

        # Standardize target x
        x_std = (x - muap_mean) / muap_std

        x_std_pred = model(c)          # (B, C, H, W)

        loss = ((x_std_pred - x_std) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = x.size(0)
        running_loss += loss.item() * B
        n_samples += B

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: BioMimeRegressor,
    loader: DataLoader,
    device: torch.device,
    muap_mean: torch.Tensor,
    muap_std: torch.Tensor,
) -> Tuple[float, float]:
    """
    Evaluate on validation set:
      - mean MSE on standardized MUAPs
      - nRMSE on original MUAPs
    """
    model.eval()
    running_loss = 0.0
    running_nrmse = 0.0
    n_samples = 0

    muap_mean = muap_mean.to(device)
    muap_std = muap_std.to(device)

    for batch in loader:
        x = batch["muap"].to(device)
        c = batch["cond"].to(device)

        x_std = (x - muap_mean) / muap_std
        x_std_pred = model(c)

        loss = ((x_std_pred - x_std) ** 2).mean()

        # De-standardize to compute nRMSE in original scale
        x_pred = x_std_pred * muap_std + muap_mean
        nrmse = batch_nrmse(x_pred, x)

        B = x.size(0)
        running_loss += loss.item() * B
        running_nrmse += nrmse.item() * B
        n_samples += B

    mean_loss = running_loss / max(n_samples, 1)
    mean_nrmse = running_nrmse / max(n_samples, 1)
    return mean_loss, mean_nrmse


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .pt file saved with save_biomime_muap_dataset")
    parser.add_argument("--save_path", type=str, default="biomime_regressor_ckpt.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of dataset used for validation (0.0 disables validation)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_basis", type=int, default=64,
                        help="Number of temporal basis functions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load dataset
    full_dataset = StaticMuapDataset(args.data_path)  # dict with "muap": [C,H,W], "cond": [6]
    print(f"Loaded dataset from {args.data_path} with {len(full_dataset)} samples.")

    # 2) Train/Val split
    if args.val_split > 0.0:
        n_total = len(full_dataset)
        n_val = int(n_total * args.val_split)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
        print(f"Train/Val split: {n_train} / {n_val}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print("No validation set (val_split=0.0).")

    # 3) DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # 4) Compute MUAP normalization stats on training data
    print("Computing MUAP normalization stats (per electrode) on training set...")
    muap_mean, muap_std = compute_muap_normalization_stats(train_loader, device)
    print("Done. Example mean/std [first channel, first few rows]:")
    print(muap_mean[0, 0, :3, 0].cpu(), muap_std[0, 0, :3, 0].cpu())

    # 5) Infer shapes
    sample_batch = next(iter(train_loader))
    _, C, H, W = sample_batch["muap"].shape  # (B, C, H, W)

    print(W)
    # 6) Instantiate model
    # model = BioMimeRegressor(
    #     c_dim=6,
    #     channels=C,
    #     height=H,
    #     time_length=W,
    #     emb_dim=args.emb_dim,
    #     hidden_dim=args.hidden_dim,
    #     n_basis=args.n_basis,
    # ).to(device)

    # model = BioMimeCNNRegressor(
    #     c_dim=6,
    #     channels=C,
    #     height=H,
    #     time_length=W,
    #     emb_dim=128,         # you can tune
    #     hidden_channels=64,  # you can tune
    #     n_blocks=4,          # 3–6 is a good range to try
    # ).to(device)

    model = BioMimeResNetCNNRegressor(
        c_dim=6,
        channels=C,
        height=H,
        time_length=W,
        emb_dim=args.emb_dim if hasattr(args, "emb_dim") else 128,
        hidden_channels=getattr(args, "hidden_channels", 128),
        n_blocks=getattr(args, "n_blocks", 6),
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=1e-4,
    )

    best_val_nrmse = float("inf")

    # 7) Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, muap_mean, muap_std)

        if val_loader is not None:
            val_loss, val_nrmse = evaluate(model, val_loader, device, muap_mean, muap_std)
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE(std): {train_loss:.6f} | "
                f"Val MSE(std): {val_loss:.6f} | "
                f"Val nRMSE(orig): {val_nrmse:.6f}"
            )

            # Save best model by val nRMSE
            if val_nrmse < best_val_nrmse:
                best_val_nrmse = val_nrmse
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "muap_mean": muap_mean.cpu(),
                    "muap_std": muap_std.cpu(),
                    "config": vars(args),
                }
                torch.save(ckpt, args.save_path)
                print(f"  -> Saved new best model to {args.save_path}")
        else:
            print(f"Epoch {epoch:03d} | Train MSE(std): {train_loss:.6f}")

    # If no validation, save final model
    if val_loader is None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "muap_mean": muap_mean.cpu(),
            "muap_std": muap_std.cpu(),
            "config": vars(args),
        }
        torch.save(ckpt, args.save_path)
        print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()
