import torch
import torch.nn as nn
import numpy as np

from BioMime.utils.basics import update_config, load_generator

from BioMime.models.generator import Generator

from archive.muap_to_emg import *

class BiomimeMuapGenerator(nn.Module):
    def __init__(self, config_path, model_checkpoint, zi=None, device='cpu'):
        """
        Initialize the Biomime MUAP Generator.

        Args:
            config_path (str): Path to the configuration YAML file.
            model_checkpoint (str): Path to the model checkpoint file.
            zi (torch.Tensor): Fixed latent vector z. If None, a random z will be generated.
            device (str): Device to load the model on ('cpu' or 'cuda:0').
        """
        super(BiomimeMuapGenerator, self).__init__()

        self.device = device
        self.cfg = update_config(config_path)
        self.biomime = Generator(self.cfg.Model.Generator).to(self.device)
        self.biomime = load_generator(model_checkpoint, self.biomime, self.device)

        if zi is None:
            self.zi = torch.randn(1, self.cfg.Model.Generator.Latent).to(self.device).repeat([1, 1])
        else:
            self.zi = zi.to(self.device)

        self.zi.requires_grad_(False)

    def forward(self, conds):
        """
        Generate MUAPs using the Biomime conditional generator.

        Args:
            conds (torch.Tensor): Conditioning input of shape (n_mus, cond_dim).

        Returns:
            torch.Tensor: Generated MUAPs of shape (n_mus, 320, 96).
        """
        n_mus = conds.shape[0]
        # adjust conds to the correct range [0.5, 1]
        conds = conds * 0.5 + 0.5
        return self.biomime.sample(n_mus, conds, conds.device, self.zi.repeat(n_mus, 1))

    def generate_muaps(self, conds):
        """
        Generate MUAPs and convert them to NumPy array.

        Args:
            conds (torch.Tensor): Conditioning input of shape (n_mus, cond_dim).

        Returns:
            np.ndarray: Generated MUAPs of shape (n_mus, 320, 96).
        """
        muaps = self.forward(conds).reshape(conds.shape[0], 96, 320)
        return muaps.permute(0, 2, 1).detach().cpu().numpy()

    def generate_batched_muaps(self, conds, batch_size):
        """
        Generate MUAPs in batches to handle large inputs that exceed memory capacity.

        Args:
            conds (torch.Tensor): Conditioning input of shape (n_mus, cond_dim).
            batch_size (int): Number of conditions to process in each batch.

        Returns:
            np.ndarray: Generated MUAPs of shape (n_mus, 320, 96).
        """
        n_mus = conds.shape[0]
        muaps_list = []

        for i in range(0, n_mus, batch_size):
            batch_conds = conds[i:i + batch_size]
            batch_muaps = self.generate_muaps(batch_conds)
            muaps_list.append(batch_muaps)

        return np.concatenate(muaps_list, axis=0)

    @staticmethod
    def generate_power_law_conditions(conditions):
        """
        Adjust given conditions so that the motor unit size follows a power law.

        Args:
            conditions (torch.Tensor): Existing conditions of shape (n_mus, cond_dim).

        Returns:
            torch.Tensor: Adjusted conditions with MU sizes following a power law.
        """
        n_mus = conditions.shape[0]

        # Generate motor unit sizes following a power law
        mu_sizes = np.random.pareto(a=2.0, size=n_mus) + 1
        mu_sizes = torch.tensor(mu_sizes / mu_sizes.max(), dtype=torch.float32)  # Normalize sizes to [0, 1]

        # Sort the motor unit sizes
        mu_sizes, _ = torch.sort(mu_sizes, descending=True)

        conditions[:, 0] = mu_sizes  # Set the motor unit size (dimension 0)
        return conditions