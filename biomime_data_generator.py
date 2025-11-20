
# Muscle-specific ranges you gave
DEPTH = {
    'ECRB': [0.0130, 0.0220],
    'ECRL': [0.0085, 0.0153],
    'PL':   [0.0071, 0.0114],
    'FCU_u': [0.0084, 0.0168],
    'FCU_h': [0.0074, 0.0165],
    'ECU':  [0.0092, 0.0168],
    'EDI':  [0.0079, 0.0171],
    'FDSI': [0.0169, 0.0231],
    'FCU':  [0.0074, 0.0168],
}

ANGLE = {
    'ECRB': [0.4946, 0.6632],
    'ECRL': [0.4607, 0.5109],
    'PL':   [0.0540, 0.0956],
    'FCU_u': [0.7878, 0.8658],
    'FCU_h': [0.9897, 1.0],
    'ECU':  [0.7194, 0.7779],
    'EDI':  [0.5637, 0.6826],
    'FDSI': [0.1471, 0.2264],
    'FCU':  [0.7878, 1.0],
}


import math
import itertools
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

# --- BioMime discrete generative factors (from the paper) ---
NUM_VALUES     = (200.0, 266.0, 333.0, 400.0)   # fibre densities
CV_VALUES      = (3.0, 3.5, 4.0, 4.5)           # m/s
IZ_VALUES      = (0.4, 0.46, 0.53, 0.6)         # relative position
LENGTH_VALUES  = (0.85, 0.95, 1.05, 1.15)       # relative fibre length

@dataclass
class BioMimeParamStats:
    """Holds global min/max for each of the 6 parameters."""
    param_min: np.ndarray  # shape (6,)
    param_max: np.ndarray  # shape (6,)
    # order: [num, depth, angle, iz, cv, length]


def compute_global_param_stats(
    depth_ranges: Dict[str, Sequence[float]],
    angle_ranges: Dict[str, Sequence[float]],
) -> BioMimeParamStats:
    """Compute global min/max for normalisation to [0.5, 1.0]."""
    depth_min = min(v[0] for v in depth_ranges.values())
    depth_max = max(v[1] for v in depth_ranges.values())
    angle_min = min(v[0] for v in angle_ranges.values())
    angle_max = max(v[1] for v in angle_ranges.values())

    num_min, num_max = min(NUM_VALUES), max(NUM_VALUES)
    iz_min, iz_max   = min(IZ_VALUES), max(IZ_VALUES)
    cv_min, cv_max   = min(CV_VALUES), max(CV_VALUES)
    len_min, len_max = min(LENGTH_VALUES), max(LENGTH_VALUES)

    param_min = np.array(
        [num_min, depth_min, angle_min, iz_min, cv_min, len_min],
        dtype=np.float32,
    )
    param_max = np.array(
        [num_max, depth_max, angle_max, iz_max, cv_max, len_max],
        dtype=np.float32,
    )
    return BioMimeParamStats(param_min=param_min, param_max=param_max)


class BioMimeConditionSampler:
    """
    Samples BioMime-style 6D condition vectors:

        [num, depth, angle, iz, cv, length]

    - num, iz, cv, length are from the original 4^4 grid.
    - depth, angle depend on a muscle chosen from DEPTH/ANGLE dicts.
    """

    def __init__(
        self,
        n_samples: int,
        depth_ranges: Dict[str, Sequence[float]] = DEPTH,
        angle_ranges: Dict[str, Sequence[float]] = ANGLE,
        seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples
        self.depth_ranges = depth_ranges
        self.angle_ranges = angle_ranges
        self.muscle_names: List[str] = sorted(depth_ranges.keys())
        self.rng = np.random.default_rng(seed)
        self.stats = compute_global_param_stats(depth_ranges, angle_ranges)

        self.raw_params, self.cond_norm, self.muscle_ids = self._build()

    def _build(self):
        """
        Returns:
            raw_params: (N, 6) unnormalized physical parameters
            cond_norm: (N, 6) normalized to [0.5, 1]
            muscle_ids: (N,) int indices into self.muscle_names
        """
        # 1) Build 4^4 grid of [num, iz, cv, length]
        combos = np.array(
            list(itertools.product(NUM_VALUES, IZ_VALUES, CV_VALUES, LENGTH_VALUES)),
            dtype=np.float32,
        )  # shape (256, 4)

        n_combo = combos.shape[0]
        reps = math.ceil(self.n_samples / n_combo)
        tiled = np.tile(combos, (reps, 1))  # (reps * 256, 4)

        # Shuffle and truncate to n_samples
        self.rng.shuffle(tiled, axis=0)
        tiled = tiled[: self.n_samples]
        num = tiled[:, 0]
        iz = tiled[:, 1]
        cv = tiled[:, 2]
        length = tiled[:, 3]

        # 2) Assign muscles ~balanced
        n_muscles = len(self.muscle_names)
        base = self.n_samples // n_muscles
        remainder = self.n_samples % n_muscles

        counts = np.full(n_muscles, base, dtype=np.int64)
        counts[:remainder] += 1

        muscle_ids = np.concatenate(
            [np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]
        )
        self.rng.shuffle(muscle_ids)
        assert muscle_ids.shape[0] == self.n_samples

        # 3) Sample depth/angle within each muscle range
        depth = np.empty(self.n_samples, dtype=np.float32)
        angle = np.empty(self.n_samples, dtype=np.float32)

        for i, mid in enumerate(muscle_ids):
            m = self.muscle_names[mid]
            d_lo, d_hi = self.depth_ranges[m]
            a_lo, a_hi = self.angle_ranges[m]
            depth[i] = self.rng.uniform(d_lo, d_hi)
            angle[i] = self.rng.uniform(a_lo, a_hi)

        # 4) Stack into raw parameter matrix
        raw_params = np.stack(
            [num, depth, angle, iz, cv, length],
            axis=1,  # (N, 6)
        ).astype(np.float32)

        # 5) Normalize to [0.5, 1] (as in the paper)
        pmin = self.stats.param_min[None, :]  # (1, 6)
        pmax = self.stats.param_max[None, :]
        cond_norm = 0.5 + 0.5 * (raw_params - pmin) / (pmax - pmin)

        return raw_params, cond_norm, muscle_ids


class BioMimeMuapDataset(Dataset):
    """
    Pytorch Dataset that:
      - samples BioMime-style conditions
      - uses your trained BioMime generator to produce MUAPs

    Returns per __getitem__:
        {
          "muap":      Tensor [H, W, T]  (here 10,32,96),
          "cond":      Tensor [6]        (normalized [0.5,1]),
          "cond_raw":  Tensor [6]        (physical units),
          "muscle_id": int,
          "muscle":    str,
        }
    """

    def __init__(
        self,
        generator,                 # your trained BioMime Generator
        n_samples: int,
        zi: Optional[torch.Tensor] = None,  # fixed z, or None to let generator handle it
        device: Optional[torch.device] = None,
        depth_ranges: Dict[str, Sequence[float]] = DEPTH,
        angle_ranges: Dict[str, Sequence[float]] = ANGLE,
        seed: Optional[int] = None,
        cache_muaps: bool = False,
        dtype: torch.dtype = torch.float32,
        batch_gen_size: int = 64,  # for precomputation
    ) -> None:
        super().__init__()

        self.generator = generator
        self.generator.eval()
        self.dtype = dtype
        self.device = (
            device
            if device is not None
            else next(generator.parameters()).device
        )
        self.zi_template = zi
        self.cache_muaps = cache_muaps
        self.batch_gen_size = batch_gen_size

        # Sample parameters
        self.sampler = BioMimeConditionSampler(
            n_samples=n_samples,
            depth_ranges=depth_ranges,
            angle_ranges=angle_ranges,
            seed=seed,
        )

        # Store as tensors
        self.cond_norm = torch.from_numpy(self.sampler.cond_norm).to(dtype)
        self.cond_raw = torch.from_numpy(self.sampler.raw_params).to(dtype)
        self.muscle_ids = torch.from_numpy(self.sampler.muscle_ids).long()
        self.muscle_names = self.sampler.muscle_names

        # Optionally pre-generate all MUAPs
        self.muaps = None
        if cache_muaps:
            self.muaps = self._generate_all_muaps()

    def __len__(self) -> int:
        return self.cond_norm.shape[0]

    def materialize_muaps(self) -> torch.Tensor:
        """
        Ensure that self.muaps is filled with [N, H, W, T] on CPU.
        Returns the tensor.
        """
        if self.muaps is None:
            self.muaps = self._generate_all_muaps()
        return self.muaps


    def _make_zi(self, batch_size: int):
        """
        Expand a fixed zi to the current batch size, if provided.
        If zi_template is None, return None.
        """
        if self.zi_template is None:
            return None

        zi = self.zi_template
        if zi.dim() == 1:  # latent dim only
            zi = zi.unsqueeze(0)  # (1, latent_dim)

        if zi.size(0) == 1 and batch_size > 1:
            zi = zi.expand(batch_size, -1)

        return zi.to(self.device)

    @torch.no_grad()
    def _generate_batch_muaps(self, cond_batch: torch.Tensor) -> torch.Tensor:
        """
        cond_batch: [B, 6] on CPU
        Returns MUAPs as [B, H, W, T] on CPU (float32).
        """
        self.generator.eval()
        cond_batch = cond_batch.to(self.device)
        num_mus = cond_batch.size(0)

        zi = self._make_zi(num_mus)

        # Your generator API:
        sim = self.generator.sample(num_mus, cond_batch.float(), self.device, zi)
        # sim expected shape: [B, T, H, W]; match your permute:
        sim = sim.permute(0, 2, 3, 1)  # -> [B, H, W, T]
        sim = sim.detach().cpu().to(self.dtype)
        return sim

    @torch.no_grad()
    def _generate_all_muaps(self) -> torch.Tensor:
        """
        Precompute MUAPs for the entire dataset in batches.

        Returns:
            Tensor [N, H, W, T] on CPU.
        """
        muaps = []
        N = len(self)
        for start in range(0, N, self.batch_gen_size):
            end = min(N, start + self.batch_gen_size)
            cond_batch = self.cond_norm[start:end]
            sim = self._generate_batch_muaps(cond_batch)
            muaps.append(sim)

        return torch.cat(muaps, dim=0)

    def __getitem__(self, idx: int):
        cond = self.cond_norm[idx]      # normalized [6]
        cond_raw = self.cond_raw[idx]   # physical [6]
        muscle_id = int(self.muscle_ids[idx].item())
        muscle_name = self.muscle_names[muscle_id]

        if self.cache_muaps:
            muap = self.muaps[idx]      # [H, W, T]
        else:
            muap = self._generate_batch_muaps(cond.unsqueeze(0))[0]

        return {
            "muap": muap,              # (10, 32, 96)
            "cond": cond,              # normalized
            "cond_raw": cond_raw,      # physical
            "muscle_id": muscle_id,
            "muscle": muscle_name,
        }



import os
from pathlib import Path

def save_biomime_muap_dataset(dataset: BioMimeMuapDataset, path: str) -> None:
    """
    Save a BioMimeMuapDataset (MUAPs + parameters) to disk.

    Args:
        dataset: an instance of BioMimeMuapDataset
        path:    file path ending in .pt, e.g. "data/biomime_train.pt"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure MUAPs are materialized
    with torch.no_grad():
        muaps = dataset.materialize_muaps().cpu()       # [N, H, W, T]

    obj = {
        "muaps": muaps,                                 # [N, H, W, T]
        "cond": dataset.cond_norm.cpu(),                # [N, 6] normalized
        "cond_raw": dataset.cond_raw.cpu(),             # [N, 6] physical
        "muscle_ids": dataset.muscle_ids.cpu(),         # [N]
        "muscle_names": dataset.muscle_names,           # list[str]

        # Useful meta info if you want to reconstruct samplers, etc.
        "depth_ranges": dataset.sampler.depth_ranges,
        "angle_ranges": dataset.sampler.angle_ranges,
        "param_min": dataset.sampler.stats.param_min,
        "param_max": dataset.sampler.stats.param_max,
        "meta": {
            "n_samples": len(dataset),
            "format_version": 1,
        },
    }

    torch.save(obj, path)


class StaticMuapDataset(Dataset):
    """
    Dataset that loads a pre-generated MUAP dataset from disk.

    Expects an object saved with `save_biomime_muap_dataset`.
    """

    def __init__(self, data_path: str, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        obj = torch.load(data_path, map_location="cpu")

        self.muaps = obj["muaps"].to(dtype)        # [N, H, W, T]
        self.cond = obj["cond"].to(dtype)          # [N, 6] normalized
        self.cond_raw = obj["cond_raw"].to(dtype)  # [N, 6] physical
        self.muscle_ids = obj["muscle_ids"].long() # [N]
        self.muscle_names = obj["muscle_names"]    # list[str]

        self.depth_ranges = obj.get("depth_ranges", None)
        self.angle_ranges = obj.get("angle_ranges", None)
        self.param_min = obj.get("param_min", None)
        self.param_max = obj.get("param_max", None)
        self.meta = obj.get("meta", {})

    def __len__(self) -> int:
        return self.muaps.shape[0]

    def __getitem__(self, idx: int):
        muscle_id = int(self.muscle_ids[idx].item())
        return {
            "muap": self.muaps[idx],              # [H, W, T]
            "cond": self.cond[idx],               # normalized
            "cond_raw": self.cond_raw[idx],       # physical
            "muscle_id": muscle_id,
            "muscle": self.muscle_names[muscle_id],
        }
