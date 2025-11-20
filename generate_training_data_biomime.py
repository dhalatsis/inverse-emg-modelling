import torch

from torch.utils.data import DataLoader
from biomime_data_generator import BioMimeMuapDataset,save_biomime_muap_dataset

# Assume you already have:
#   generator = your trained BioMime Generator(...)
#   generator.load_state_dict(...)
#   generator.to(device)
from biomime_generator import *
from tqdm import tqdm
cfg = update_config('./config/config.yaml')

generator = Generator(cfg.Model.Generator)
generator = load_generator('ckp/model_linear.pth', generator, 'cuda:0')


device = next(generator.parameters()).device
zi_fixed = torch.zeros(16, device=device)  # or None

for k in tqdm(range(15)):   # e.g. 10 chunks
    ds = BioMimeMuapDataset(
        generator=generator,
        n_motor_units=100,    # => 1000 * 256 samples in this chunk
        zi=zi_fixed,
        device=device,
        seed=123 + k,
        cache_muaps=False,
    )
    save_biomime_muap_dataset(ds, f"~/data_e/biomime_training_data/biomime_chunk_{k:02d}.pt")
