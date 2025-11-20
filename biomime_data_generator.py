import math
import itertools
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, List, Union
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Use your actual muscle dictionaries here
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

# Discrete generative factors (4^4 grid)
NUM_VALUES     = (200.0, 266.0, 333.0, 400.0)
CV_VALUES      = (3.0, 3.5, 4.0, 4.5)
IZ_VALUES      = (0.4, 0.46, 0.53, 0.6)
LENGTH_VALUES  = (0.85, 0.95, 1.05, 1.15)


@dataclass
class BioMimeParamStats:
    """Global min/max for each of the 6 parameters."""
    param_min: np.ndarray  # [6]
    param_max: np.ndarray  # [6]
    # order: [num, depth, angle, iz, cv, length]


def compute_global_param_stats(
    depth_ranges: Dict[str, Sequence[float]],
    angle_ranges: Dict[str, Sequence[float]],
) -> BioMimeParamStats:
    """Compute global min/max to normalize to [0.5, 1]."""
    depth_min = min(v[0] for v in depth_ranges.values())
    depth_max = max(v[1] for v in depth_ranges.values())
    angle_min = min(v[0] for v in angle_ranges.values())
    angle_max = max(v[1] for v in angle_ranges.values())

    num_min, num_max = min(NUM_VALUES), max(NUM_VALUES)
    iz_min,  iz_max  = min(IZ_VALUES), max(IZ_VALUES)
    cv_min,  cv_max  = min(CV_VALUES), max(CV_VALUES)
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
    Given N motor units:
      - Sample a muscle + (depth, angle) for each motor unit.
      - For each motor unit, create all 256 combinations of [num, iz, cv, length].
    Final number of samples: N * 256

    Produces:
      raw_params: [N*256, 6]  (physical)
      cond_norm:  [N*256, 6]  (normalized to [0.5, 1])
      muscle_ids: [N*256]     (index into muscle_names per sample)
      unit_ids:   [N*256]     (which MU this sample belongs to, 0..N-1)
    """

    def __init__(
        self,
        n_motor_units: int,
        depth_ranges: Dict[str, Sequence[float]] = DEPTH,
        angle_ranges: Dict[str, Sequence[float]] = ANGLE,
        seed: Optional[int] = None,
    ) -> None:
        self.n_motor_units = n_motor_units
        self.depth_ranges = depth_ranges
        self.angle_ranges = angle_ranges
        self.muscle_names: List[str] = sorted(depth_ranges.keys())
        self.rng = np.random.default_rng(seed)
        self.stats = compute_global_param_stats(depth_ranges, angle_ranges)

        (
            self.raw_params,
            self.cond_norm,
            self.muscle_ids,
            self.unit_ids,
        ) = self._build()

    def _build(self):
        # 256 combinations of [num, iz, cv, length]
        combos = np.array(
            list(itertools.product(NUM_VALUES, IZ_VALUES, CV_VALUES, LENGTH_VALUES)),
            dtype=np.float32,
        )  # [256, 4] in order [num, iz, cv, length]
        n_combo = combos.shape[0]  # 256

        n_units = self.n_motor_units
        n_total = n_units * n_combo

        # 1) Assign muscles per motor unit (balanced)
        n_muscles = len(self.muscle_names)
        base = n_units // n_muscles
        remainder = n_units % n_muscles

        counts = np.full(n_muscles, base, dtype=np.int64)
        counts[:remainder] += 1

        muscle_ids_per_unit = np.concatenate(
            [np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]
        )
        self.rng.shuffle(muscle_ids_per_unit)
        assert muscle_ids_per_unit.shape[0] == n_units

        # 2) Sample depth/angle for each motor unit
        depth_per_unit = np.empty(n_units, dtype=np.float32)
        angle_per_unit = np.empty(n_units, dtype=np.float32)
        for i, mid in enumerate(muscle_ids_per_unit):
            m = self.muscle_names[mid]
            d_lo, d_hi = self.depth_ranges[m]
            a_lo, a_hi = self.angle_ranges[m]
            depth_per_unit[i] = self.rng.uniform(d_lo, d_hi)
            angle_per_unit[i] = self.rng.uniform(a_lo, a_hi)

        # 3) Tile 4^4 combos across units
        tiled = np.tile(combos, (n_units, 1))  # [n_units*256, 4]
        num_all    = tiled[:, 0]
        iz_all     = tiled[:, 1]
        cv_all     = tiled[:, 2]
        length_all = tiled[:, 3]

        # 4) Expand depth/angle/muscle_id per sample
        depth_all = np.repeat(depth_per_unit, n_combo)
        angle_all = np.repeat(angle_per_unit, n_combo)
        muscle_ids_all = np.repeat(muscle_ids_per_unit, n_combo)

        # 5) Unit ID per sample (0..n_units-1)
        unit_ids_all = np.repeat(np.arange(n_units, dtype=np.int64), n_combo)

        # 6) Raw parameters in the order used by BioMime: [num, depth, angle, iz, cv, length]
        raw_params = np.stack(
            [num_all, depth_all, angle_all, iz_all, cv_all, length_all],
            axis=1,
        ).astype(np.float32)

        # 7) Normalize to [0.5, 1]
        pmin = self.stats.param_min[None, :]
        pmax = self.stats.param_max[None, :]
        cond_norm = 0.5 + 0.5 * (raw_params - pmin) / (pmax - pmin)

        assert raw_params.shape[0] == n_total
        return raw_params, cond_norm, muscle_ids_all, unit_ids_all


class BioMimeMuapDataset(Dataset):
    """
    Uses your BioMime generator to turn condition vectors into MUAPs.

    Args:
        generator:    trained BioMime Generator
        n_motor_units: N; dataset size will be N * 256
        zi:           fixed latent z (or None to let generator handle it)
        device:       torch.device for generator
        cache_muaps:  if True, precompute all MUAPs once
    """

    def __init__(
        self,
        generator,
        n_motor_units: int,
        zi: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        depth_ranges: Dict[str, Sequence[float]] = DEPTH,
        angle_ranges: Dict[str, Sequence[float]] = ANGLE,
        seed: Optional[int] = None,
        cache_muaps: bool = False,
        dtype: torch.dtype = torch.float32,
        batch_gen_size: int = 64,
    ) -> None:
        super().__init__()

        self.generator = generator
        self.generator.eval()
        self.dtype = dtype
        self.device = (
            device if device is not None else next(generator.parameters()).device
        )
        self.zi_template = zi
        self.cache_muaps = cache_muaps
        self.batch_gen_size = batch_gen_size

        # New sampler: N motor units -> N*256 samples
        self.sampler = BioMimeConditionSampler(
            n_motor_units=n_motor_units,
            depth_ranges=depth_ranges,
            angle_ranges=angle_ranges,
            seed=seed,
        )

        self.cond_norm = torch.from_numpy(self.sampler.cond_norm).to(dtype)
        self.cond_raw = torch.from_numpy(self.sampler.raw_params).to(dtype)
        self.muscle_ids = torch.from_numpy(self.sampler.muscle_ids).long()
        self.unit_ids = torch.from_numpy(self.sampler.unit_ids).long()
        self.muscle_names = self.sampler.muscle_names

        self.muaps = None
        if cache_muaps:
            self.muaps = self._generate_all_muaps()

    def __len__(self) -> int:
        # Total samples = n_motor_units * 256
        return self.cond_norm.shape[0]

    # -------- MUAP generation helpers --------

    def materialize_muaps(self) -> torch.Tensor:
        """
        Ensure that self.muaps is filled with [N, H, W, T] on CPU.
        """
        if self.muaps is None:
            self.muaps = self._generate_all_muaps()
        return self.muaps

    def _make_zi(self, batch_size: int):
        if self.zi_template is None:
            return None
        zi = self.zi_template
        if zi.dim() == 1:
            zi = zi.unsqueeze(0)  # [1, latent_dim]
        if zi.size(0) == 1 and batch_size > 1:
            zi = zi.expand(batch_size, -1)
        return zi.to(self.device)

    @torch.no_grad()
    def _generate_batch_muaps(self, cond_batch: torch.Tensor) -> torch.Tensor:
        """
        cond_batch: [B, 6] on CPU
        Returns MUAPs: [B, H, W, T] on CPU (float32).
        """
        self.generator.eval()
        cond_batch = cond_batch.to(self.device)
        num_mus = cond_batch.size(0)
        zi = self._make_zi(num_mus)

        # Your existing generator API:
        # sim: [B, T, H, W]
        sim = self.generator.sample(num_mus, cond_batch.float(), self.device, zi)

        # Match your own permutation: [B, H, W, T]
        sim = sim.permute(0, 2, 3, 1).detach().cpu().to(self.dtype)
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

    # -------- Dataset API --------

    def __getitem__(self, idx: int):
        cond = self.cond_norm[idx]      # [6], normalized
        cond_raw = self.cond_raw[idx]   # [6], physical
        muscle_id = int(self.muscle_ids[idx].item())
        muscle_name = self.muscle_names[muscle_id]
        unit_id = int(self.unit_ids[idx].item())

        if self.cache_muaps:
            muap = self.muaps[idx]      # [H, W, T]
        else:
            muap = self._generate_batch_muaps(cond.unsqueeze(0))[0]

        return {
            "muap": muap,              # [10, 32, 96]
            "cond": cond,
            "cond_raw": cond_raw,
            "muscle_id": muscle_id,
            "muscle": muscle_name,
            "unit_id": unit_id,        # which motor unit this sample came from
        }


def save_biomime_muap_dataset(dataset: BioMimeMuapDataset, path: str) -> None:
    """
    Save MUAPs + parameters to a single .pt file.
    Can be later loaded (and merged) by StaticMuapDataset.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure MUAPs exist on CPU
    with torch.no_grad():
        muaps = dataset.materialize_muaps().cpu()

    obj = {
        "muaps": muaps,                         # [N, H, W, T]
        "cond": dataset.cond_norm.cpu(),        # [N, 6] normalized
        "cond_raw": dataset.cond_raw.cpu(),     # [N, 6] physical
        "muscle_ids": dataset.muscle_ids.cpu(), # [N]
        "muscle_names": dataset.muscle_names,   # list[str]

        "depth_ranges": dataset.sampler.depth_ranges,
        "angle_ranges": dataset.sampler.angle_ranges,
        "param_min": dataset.sampler.stats.param_min,
        "param_max": dataset.sampler.stats.param_max,

        "meta": {
            "n_samples": len(dataset),                # = n_motor_units * 256
            "n_motor_units": dataset.sampler.n_motor_units,
            "format_version": 2,
        },
    }

    # Include unit_ids if present
    if getattr(dataset, "unit_ids", None) is not None:
        obj["unit_ids"] = dataset.unit_ids.cpu()

    torch.save(obj, path)


class StaticMuapDataset(Dataset):
    """
    Load one or more pre-generated MUAP datasets from .pt files
    (saved by save_biomime_muap_dataset) and present them as a
    single PyTorch Dataset.

    Args:
        data_paths: str or list[str] of .pt files.
    """

    def __init__(
        self,
        data_paths: Union[str, Sequence[str]],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Normalize to list of paths
        if isinstance(data_paths, (str, Path)):
            paths = [str(data_paths)]
        else:
            paths = [str(p) for p in data_paths]

        muaps_list = []
        cond_list = []
        cond_raw_list = []
        muscle_ids_list = []
        unit_ids_list = []

        self.muscle_names = None
        self.depth_ranges = None
        self.angle_ranges = None
        self.param_min = None
        self.param_max = None
        self.source_metas = []

        for i, path in enumerate(paths):
            obj = torch.load(path, map_location="cpu")

            muaps_list.append(obj["muaps"].to(dtype))
            cond_list.append(obj["cond"].to(dtype))
            cond_raw_list.append(obj["cond_raw"].to(dtype))
            muscle_ids_list.append(obj["muscle_ids"].long())
            self.source_metas.append(obj.get("meta", {}))

            if i == 0:
                # First file sets the reference
                self.muscle_names = obj["muscle_names"]
                self.depth_ranges = obj.get("depth_ranges", None)
                self.angle_ranges = obj.get("angle_ranges", None)
                self.param_min = obj.get("param_min", None)
                self.param_max = obj.get("param_max", None)

                if "unit_ids" in obj:
                    unit_ids_list.append(obj["unit_ids"].long())
            else:
                # Check compatibility
                if obj["muscle_names"] != self.muscle_names:
                    raise ValueError("Muscle name mismatch across dataset files")

                # Handle unit_ids presence/absence
                if "unit_ids" in obj:
                    if unit_ids_list:
                        unit_ids_list.append(obj["unit_ids"].long())
                    else:
                        # Previous files had no unit_ids: pad them with -1
                        unit_ids_list.append(
                            torch.full_like(muscle_ids_list[-1], fill_value=-1)
                        )
                else:
                    if unit_ids_list:
                        # Previous files had unit_ids; pad this one with -1
                        unit_ids_list.append(
                            torch.full_like(muscle_ids_list[-1], fill_value=-1)
                        )

        # Concatenate across all files
        self.muaps = torch.cat(muaps_list, dim=0)      # [sumN, H, W, T]
        self.cond = torch.cat(cond_list, dim=0)        # [sumN, 6]
        self.cond_raw = torch.cat(cond_raw_list, dim=0)# [sumN, 6]
        self.muscle_ids = torch.cat(muscle_ids_list, dim=0) # [sumN]

        if unit_ids_list:
            self.unit_ids = torch.cat(unit_ids_list, dim=0).long()
        else:
            self.unit_ids = None

        self.meta = {
            "total_samples": self.muaps.shape[0],
            "source_files": paths,
            "source_metas": self.source_metas,
        }

    def __len__(self) -> int:
        return self.muaps.shape[0]

    def __getitem__(self, idx: int):
        muscle_id = int(self.muscle_ids[idx].item())
        item = {
            "muap": self.muaps[idx],               # [H, W, T]
            "cond": self.cond[idx],                # [6]
            "cond_raw": self.cond_raw[idx],        # [6]
            "muscle_id": muscle_id,
            "muscle": self.muscle_names[muscle_id],
        }
        if self.unit_ids is not None:
            item["unit_id"] = int(self.unit_ids[idx].item())
        return item
