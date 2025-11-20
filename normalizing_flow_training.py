"""
train_biomime_flow.py

Train a conditional normalizing flow decoder on a static BioMime MUAP dataset.

- Uses your StaticMuapDataset, which loads a .pt file created by
  save_biomime_muap_dataset(..).

- Model: FlowDecoder
    Input:  x_std  ~ standardized MUAP, shape (B, C=10, H=32, W=96)
            c      ~ normalized 6D BioMime params (in [0.5, 1])
    Output: NLL on x_std, exact inverse x_std <-> z

- Metrics:
    * Train / Val NLL
    * Val reconstruction nRMSE in ORIGINAL MUAP scale
"""

import math
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from biomime_data_generator import StaticMuapDataset


# ------------------------------
# Metrics
# ------------------------------

def batch_nrmse(x_pred: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute batch-wise normalized RMSE:
      nRMSE = RMSE(x_pred - x_true) / RMS(x_true)
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
# Conditioning MLP
# ------------------------------

class ConditionMLP(nn.Module):
    """
    Maps 6D condition vector -> embedding used in all coupling layers.
    """

    def __init__(self, in_dim: int = 6, emb_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        c: (B, 6) in [0.5, 1.0]
        returns: (B, emb_dim)
        """
        return self.net(c)


# ------------------------------
# Conditioner (conv + cond embedding concat)
# ------------------------------

class ConvConditioner(nn.Module):
    """
    CNN that predicts scale and shift fields (s, t) for affine coupling.

    Input: x_a (B, C, H, W), cond_emb (B, cond_dim)
    Output: s, t (B, C, H, W)
    """

    def __init__(self, in_channels: int, cond_dim: int, hidden_channels: int = 32):
        super().__init__()
        self.cond_dim = cond_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels + cond_dim, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(hidden_channels, 2 * in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_a: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_a:      (B, C, H, W)
        cond_emb: (B, cond_dim)
        returns:  s, t each (B, C, H, W)
        """
        B, C, H, W = x_a.shape
        cond = cond_emb.unsqueeze(-1).unsqueeze(-1)           # (B, cond_dim, 1, 1)
        cond = cond.expand(-1, self.cond_dim, H, W)           # (B, cond_dim, H, W)

        h = torch.cat([x_a, cond], dim=1)
        out = self.net(h)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * 0.9  # keep scale moderate
        return s, t


# ------------------------------
# Affine coupling (RealNVP-style)
# ------------------------------

class AffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling with channel-wise mask.

    x = x_a + x_b
    y_a = x_a
    y_b = x_b * exp(s) + t
    logdet = sum(s over active channels/spatial dims)
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        hidden_channels: int = 32,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert channels > 1, "Need at least 2 channels for coupling."

        if mask is None:
            mask = torch.zeros(1, channels, 1, 1)
            mask[:, : channels // 2, :, :] = 1.0

        self.register_buffer("mask", mask)  # (1, C, 1, 1)
        self.channels = channels
        self.cond_dim = cond_dim

        self.conditioner = ConvConditioner(
            in_channels=channels,
            cond_dim=cond_dim,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transform x -> y.

        x:        (B, C, H, W)
        cond_emb: (B, cond_dim)
        returns:  y, logdet (B,)
        """
        B, C, H, W = x.shape
        mask = self.mask

        x_a = x * mask
        x_b = x * (1.0 - mask)

        s, t = self.conditioner(x_a, cond_emb)  # (B, C, H, W)
        s = s * (1.0 - mask)
        t = t * (1.0 - mask)

        y_b = x_b * torch.exp(s) + t
        y = x_a + y_b

        logdet = s.view(B, -1).sum(dim=1)
        return y, logdet

    def inverse(self, y: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transform y -> x.

        y:        (B, C, H, W)
        cond_emb: (B, cond_dim)
        returns:  x, logdet_inverse (B,)
        """
        B, C, H, W = y.shape
        mask = self.mask

        y_a = y * mask
        y_b = y * (1.0 - mask)

        s, t = self.conditioner(y_a, cond_emb)
        s = s * (1.0 - mask)
        t = t * (1.0 - mask)

        x_b = (y_b - t) * torch.exp(-s)
        x = y_a + x_b

        logdet = -s.view(B, -1).sum(dim=1)
        return x, logdet


# ------------------------------
# Optional time-scaling bijector (off by default)
# ------------------------------

class TimeScalingBijector(nn.Module):
    """
    Approximate invertible time warp along last dimension (W = time).
    Used only if you enable use_time_warp in FlowDecoder.
    """

    def __init__(self, time_length: int = 96, cond_dim: int = 64):
        super().__init__()
        self.time_length = time_length
        self.fc = nn.Linear(cond_dim, 2)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        """
        x:        (B, C, H, W)
        cond_emb: (B, cond_dim)
        reverse:  False -> forward warp, True -> approx inverse
        """
        B, C, H, W = x.shape
        assert W == self.time_length, "Time dimension mismatch."

        alpha_raw, delta_raw = self.fc(cond_emb).chunk(2, dim=1)
        alpha = 0.3 * torch.tanh(alpha_raw) + 1.0        # in ~[0.7,1.3]
        delta = 0.25 * W * torch.tanh(delta_raw)         # in [-W/4, W/4]

        alpha = alpha.view(B, 1, 1, 1)
        delta = delta.view(B, 1, 1, 1)

        t = torch.linspace(0, W - 1, W, device=x.device, dtype=x.dtype)
        t = t.view(1, 1, 1, W)

        if not reverse:
            t_new = alpha * t + delta
        else:
            t_new = (t - delta) / (alpha + 1e-6)

        t_new = t_new.clamp(0, W - 1)

        x_norm = (t_new / (W - 1)) * 2.0 - 1.0
        x_norm = x_norm.expand(B, 1, H, W)

        y_vals = torch.linspace(0, H - 1, H, device=x.device, dtype=x.dtype)
        y_norm = (y_vals / (H - 1)) * 2.0 - 1.0
        y_norm = y_norm.view(1, 1, H, 1).expand(B, 1, H, W)

        grid = torch.stack([x_norm, y_norm], dim=-1).squeeze(1)  # (B, H, W, 2)

        y = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return y


# ------------------------------
# FlowDecoder
# ------------------------------

class FlowDecoder(nn.Module):
    """
    Conditional normalizing flow (decoder side):

        x_std = f_theta(z, c)
        z ~ N(0, I), same shape as x_std

    inverse(x_std, c) returns z and logdet_inverse for NLL.
    """

    def __init__(
        self,
        channels: int = 10,
        height: int = 32,
        width: int = 96,
        n_blocks: int = 8,
        hidden_channels: int = 32,
        cond_dim: int = 64,
        use_time_warp: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.use_time_warp = use_time_warp

        self.cond_mlp = ConditionMLP(in_dim=6, emb_dim=cond_dim)

        masks = []
        for i in range(n_blocks):
            mask = torch.zeros(1, channels, 1, 1)
            if i % 2 == 0:
                mask[:, : channels // 2] = 1.0
            else:
                mask[:, channels // 2 :] = 1.0
            masks.append(mask)

        self.couplings = nn.ModuleList([
            AffineCoupling(
                channels=channels,
                cond_dim=cond_dim,
                hidden_channels=hidden_channels,
                mask=masks[i],
            )
            for i in range(n_blocks)
        ])

        self.time_warp = TimeScalingBijector(time_length=width, cond_dim=cond_dim) if use_time_warp else None

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: base -> data

        z: (B, C, H, W)
        c: (B, 6)
        returns: x_std, logdet_forward (B,)
        """
        assert z.shape[1:] == (self.channels, self.height, self.width), "z shape mismatch"
        cond_emb = self.cond_mlp(c)

        x = z
        logdet = torch.zeros(z.size(0), device=z.device)

        for coupling in self.couplings:
            x, ld = coupling(x, cond_emb)
            logdet = logdet + ld

        if self.time_warp is not None:
            x = self.time_warp(x, cond_emb, reverse=False)

        return x, logdet

    def inverse(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse: data -> base

        x: (B, C, H, W)
        c: (B, 6)
        returns: z, logdet_inverse (B,)
        """
        assert x.shape[1:] == (self.channels, self.height, self.width), "x shape mismatch"
        cond_emb = self.cond_mlp(c)

        if self.time_warp is not None:
            x = self.time_warp(x, cond_emb, reverse=True)

        z = x
        logdet = torch.zeros(x.size(0), device=x.device)

        for coupling in reversed(self.couplings):
            z, ld = coupling.inverse(z, cond_emb)
            logdet = logdet + ld

        return z, logdet


# ------------------------------
# NLL on standardized MUAPs
# ------------------------------

def compute_nll(
    model: FlowDecoder,
    x_std: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Compute negative log-likelihood on standardized x_std.

    x_std: (B, C, H, W)
    c:     (B, 6)
    """
    z, logdet_inv = model.inverse(x_std, c)   # z = f^{-1}(x_std, c)

    B = x_std.size(0)
    z_flat = z.view(B, -1)
    log_p_z = -0.5 * (z_flat.pow(2).sum(dim=1) + z_flat.size(1) * math.log(2 * math.pi))

    log_p_x = log_p_z + logdet_inv
    nll = -log_p_x.mean()
    return nll


# ------------------------------
# Compute per-channel MUAP normalization stats
# ------------------------------

@torch.no_grad()
def compute_muap_normalization_stats(
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean/std for MUAPs over the given loader.

    Returns:
        muap_mean: (1, C, 1, 1)
        muap_std:  (1, C, 1, 1)
    """
    sum_ = None
    sum_sq = None
    count = 0

    for batch in loader:
        x = batch["muap"].to(device)  # (B, C, H, W)
        B, C, H, W = x.shape

        if sum_ is None:
            sum_ = torch.zeros(C, device=device, dtype=torch.float64)
            sum_sq = torch.zeros(C, device=device, dtype=torch.float64)

        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        sum_ += x_flat.sum(dim=(0, 2))
        sum_sq += (x_flat ** 2).sum(dim=(0, 2))
        count += x_flat.size(0) * x_flat.size(2)  # total pixels per channel

    mean = (sum_ / count).to(torch.float32)         # (C,)
    var = (sum_sq / count - sum_ ** 2 / (count ** 2)).to(torch.float32)
    std = torch.sqrt(var + 1e-8)

    muap_mean = mean.view(1, C, 1, 1)
    muap_std = std.view(1, C, 1, 1)
    return muap_mean, muap_std


# ------------------------------
# Training & evaluation loops
# ------------------------------

def train_one_epoch(
    model: FlowDecoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    muap_mean: torch.Tensor,
    muap_std: torch.Tensor,
) -> float:
    model.train()
    running_nll = 0.0
    n_samples = 0

    muap_mean = muap_mean.to(device)
    muap_std = muap_std.to(device)

    for batch in loader:
        x = batch["muap"].to(device)   # (B, C, H, W), here C=10
        c = batch["cond"].to(device)   # (B, 6) in [0.5,1]

        x_std = (x - muap_mean) / muap_std

        optimizer.zero_grad()
        nll = compute_nll(model, x_std, c)
        nll.backward()
        optimizer.step()

        B = x.size(0)
        running_nll += nll.item() * B
        n_samples += B

    return running_nll / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: FlowDecoder,
    loader: DataLoader,
    device: torch.device,
    muap_mean: torch.Tensor,
    muap_std: torch.Tensor,
) -> Tuple[float, float]:
    """
    Evaluate on validation loader:
      - mean NLL
      - reconstruction nRMSE in original MUAP scale
    """
    model.eval()
    running_nll = 0.0
    running_nrmse = 0.0
    n_samples = 0

    muap_mean = muap_mean.to(device)
    muap_std = muap_std.to(device)

    for batch in loader:
        x = batch["muap"].to(device)
        c = batch["cond"].to(device)

        x_std = (x - muap_mean) / muap_std

        # NLL
        nll = compute_nll(model, x_std, c)
        running_nll += nll.item() * x.size(0)

        # Round-trip recon
        z, _ = model.inverse(x_std, c)
        x_std_hat, _ = model.forward(z, c)
        x_hat = x_std_hat * muap_std + muap_mean

        nrmse = batch_nrmse(x_hat, x)
        running_nrmse += nrmse.item() * x.size(0)

        n_samples += x.size(0)

    mean_nll = running_nll / max(n_samples, 1)
    mean_nrmse = running_nrmse / max(n_samples, 1)
    return mean_nll, mean_nrmse


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .pt file saved with save_biomime_muap_dataset")
    parser.add_argument("--save_path", type=str, default="biomime_flow_ckpt.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of dataset used for validation (0.0 disables validation)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_blocks", type=int, default=8)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--use_time_warp", action="store_true",
                        help="Enable approximate invertible time scaling bijector")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load dataset
    full_dataset = StaticMuapDataset(args.data_path)  # returns dict w/ "muap": [H,W,T], "cond": [6]
    print(f"Loaded dataset from {args.data_path} with {len(full_dataset)} samples.")

    # Ensure MUAP is [C,H,W]; StaticMuapDataset gives [H,W,T] and DataLoader will stack as (B,H,W,T).
    # We interpret H as channels, W and T as spatial dims, so no permutation needed.

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

    # 4) Compute MUAP normalization stats on *train* loader
    print("Computing MUAP normalization stats on training set...")
    muap_mean, muap_std = compute_muap_normalization_stats(train_loader, device)
    print("Done. Example per-channel mean/std (first 3 channels):")
    print(muap_mean[0, :3, 0, 0].cpu(), muap_std[0, :3, 0, 0].cpu())

    # 5) Instantiate model
    _, C, H, W = next(iter(train_loader))["muap"].shape  # (B, C, H, W)
    model = FlowDecoder(
        channels=C,
        height=H,
        width=W,
        n_blocks=args.n_blocks,
        hidden_channels=args.hidden_channels,
        cond_dim=64,
        use_time_warp=args.use_time_warp,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)

    best_val_nll = float("inf")

    # 6) Training loop
    for epoch in range(1, args.epochs + 1):
        train_nll = train_one_epoch(model, train_loader, optimizer, device, muap_mean, muap_std)

        if val_loader is not None:
            val_nll, val_nrmse = evaluate(model, val_loader, device, muap_mean, muap_std)
            print(
                f"Epoch {epoch:03d} | "
                f"Train NLL: {train_nll:.4f} | "
                f"Val NLL: {val_nll:.4f} | "
                f"Val nRMSE (recon): {val_nrmse:.4f}"
            )

            # Save best model by Val NLL
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "muap_mean": muap_mean.cpu(),
                    "muap_std": muap_std.cpu(),
                    "config": vars(args),
                }
                torch.save(ckpt, args.save_path)
                print(f"  -> Saved new best model to {args.save_path}")
        else:
            print(f"Epoch {epoch:03d} | Train NLL: {train_nll:.4f}")

    # If no validation, still save final model
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
