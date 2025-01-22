import argparse
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps
from BioMime.utils.params import num_mus, steps, tgt_params
from BioMime.models.generator import Generator


cfg = update_config('./config/config.yaml')

generator = Generator(cfg.Model.Generator)
generator = load_generator('ckp/model_linear.pth', generator, 'cuda:0')


# here we generate muaps for N MUs of the same muscle, thus fixing the z
n_mus = 20

zi = torch.randn(1, cfg.Model.Generator.Latent).repeat([n_mus, 1])


cond = torch.rand(n_mus, 6)
sim = generator.sample(n_mus, cond.float(), cond.device, zi)

import torch
import torch.nn as nn
import torch.optim as optim

# Optimization Model to find the optimal input
class ConditionOptimizer(nn.Module):
    def __init__(self, biomime, initial_input, zi, device='cuda:0'):
        super(ConditionOptimizer, self).__init__()
        self.biomime = biomime # Pretrained BioMime network (not trainable here)
        self.biomime = self.biomime.to(device)
        self.opt_params = nn.Parameter(initial_input.clone().detach().requires_grad_(True)) # Optimizable input
        self.zi = zi.to(device)
        self.zi.requires_grad_(False)

    def forward(self):
        # Pass the input to the BioMime network
        output = generator.sample(n_mus, self.opt_params, self.opt_params.device, self.zi)
        return output


model = ConditionOptimizer(generator, torch.rand(n_mus, 6), zi)


def optimize_input(model,  target_output, device='cuda:0', lr=0.01, steps=1000):
    """
    Optimize the input to the BioMime network to match a target output.
    
    Args:
        biomime: Pretrained BioMime model.
        initial_input: Initial guess for the 6D input (torch Tensor).
        target_output: Desired output to optimize for.
        lr: Learning rate.
        steps: Number of optimization steps.

    Returns:
        Optimized input tensor.
    """
    # Initialize the optimizer model
    model = model.to(device)
    target_output = target_output.to(device)
    target_output.requires_grad = False
    optimizer = optim.Adam([model.opt_params], lr=lr)
    criterion = nn.MSELoss()  # Loss function

    # Optimization loop
    for step in range(steps):
        print(step)
        optimizer.zero_grad()
        output = model.forward()
        loss = criterion(output, target_output)
        loss.backward()
        optimizer.step()

        # Print progress
        if (step + 1) % 100 == 0:
            print(f"Step [{step+1}/{steps}], Loss: {loss.item():.4f}, Input: {model.input.data}")

    print("Optimization finished!")
    return model.input.detach()

optimize_input(model, sim , device='cpu')