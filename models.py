import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from time import time
from tqdm import tqdm

from emg_processor import EMGProcessor

class ConditionBiomime(nn.Module):
    def __init__(self, biomime, zi, device='cuda:0'):
        super(ConditionBiomime, self).__init__()

        self.biomime = biomime
        self.biomime = self.biomime.to(device)
        self.zi = zi.to(device)
        self.zi.requires_grad_(False)


    def forward(self, conds):
        n_mus = conds.shape[0]

        return self.biomime.sample(n_mus, conds, conds.device, self.zi.repeat(n_mus, 1))


class ConditionOptimizerLegacy(nn.Module):
    def __init__(
        self, 
        biomime, # BiomimeMuapGenerator class
        initial_input, 
        indices=[0,1,2,3,4,5], 
        margin=1.1,
        device='cuda:0'
    ):
        super(ConditionOptimizerLegacy, self).__init__()
        self.device = device
        self.biomime_muap_gen = biomime.to(device)  # Pretrained BioMime network

        self.indices = indices
        self.margin = margin

        # Make the portion of input that is learnable an nn.Parameter
        # We use logit of the initial_input so we can pass it through a sigmoid
        # self.opt_params = nn.Parameter(
        #     torch.logit(initial_input.clone().detach()[:, indices] / self.margin)
        # )
        self.opt_params = nn.Parameter(
            (initial_input.clone().detach()[:, indices] / self.margin)
        )

        # The rest of the input, which remains fixed, also store in self
        # but *not* as a parameter
        self.biomime_input = (initial_input.clone().detach().to(device))

        # Some additional parameter or condition (zi)

        # Sigmoid to map logits -> [0, 1] range
        self.sig = nn.Sigmoid()

    def forward(self, n_mus):
        """
        Forward pass: 
        1. Copies biomime_input
        2. Replaces specified indices with self.opt_params (learnable)
        3. Calls self.biomime.sample(...) to generate waveforms
        """
        biomime_input = self.biomime_input.clone()
        biomime_input[:, self.indices] = self.opt_params  # replace the chosen indices

        # We assume biomime.sample(...) returns waveforms or MUAPs
        # Make sure you pass the *sigmoid* of biomime_input if your model expects values in [0,1]
        output = self.biomime_muap_gen((biomime_input)* self.margin)
        return output

    def get_conditions(self):
        """
        Return the current conditions (i.e., the parameters in [0, 1]).
        This can be used to check the learned values after optimization.
        """
        return (self.opt_params) * self.margin


def torch_kurtosis(x: torch.Tensor) -> torch.Tensor:
    """
    Compute excess kurtosis of a 1D tensor in PyTorch.
    """
    #x = x.flatten()
    n = x.shape[1]
    mean = torch.mean(x, axis=1, keepdim=True)
    diff = x - mean
    m2 = torch.mean(diff ** 2, axis=1)
    m4 = torch.mean(diff ** 4, axis=1)
    kurt = m4 / (m2 ** 2) - 3
    return kurt * 1e-2


def torch_logcosh(x: torch.Tensor) -> torch.Tensor:
    """
    Compute excess logcosh of a 1D tensor in PyTorch.
    """
    #x = x.flatten()
    n = x.shape[1]
    mean = torch.mean(x, axis=1, keepdim=True)
    diff = x - mean
    m2 = torch.mean(diff ** 2, axis=1, keepdim=True)
    diff = diff / torch.sqrt(m2 + 1e-9)
    cosh = torch.cosh(diff.to(torch.float64))
    logcosh = torch.log(cosh) /(m2 + 1e-9)

    return logcosh.to(torch.float32).mean()


class NegentropyLoss(torch.nn.Module):
    def __init__(self):
        super(NegentropyLoss, self).__init__()

    def forward(self, y):
        # Enforce input y is zero-mean and unit variance
        y = (y - y.mean(dim=1, keepdim=True)) / (y.std(dim=1, keepdim=True) + 1e-9)
        # Log-cosh contrast function
        G_y_logcosh = torch.log(torch.cosh(y))
        G_v_logcosh = torch.log(torch.cosh(torch.randn_like(y)))

        # Square contrast function
        # negentropy = torch.mean(torch.square(y)) - torch.log(torch.cosh(torch.randn_like(y)))
        
        # Exponential contrast function
        G_y_exponential = -torch.exp(-y**2 / 2)
        G_v_exponential = -torch.exp(-torch.randn_like(y)**2 / 2)
        
        # Combine both contrast functions for negentropy
        negentropy_logcosh = torch.mean(G_y_logcosh, dim=1) - torch.mean(G_v_logcosh, dim=1)
        negentropy_exponential = torch.mean(G_y_exponential, dim=1) - torch.mean(G_v_exponential, dim=1)
        
        # Sum both to form the final combined negentropy
        negentropy = negentropy_logcosh**2 + negentropy_exponential**2
        
        # Return the negative to make this a loss function (minimize -J(y))
        return -negentropy
    



class ConditionOptimizer(nn.Module):
    def __init__(
        self,
        biomime,                # Pretrained BioMimeMuapGenerator (or similar)
        device='cuda:0',
        margin=1.0,
        indices=[0, 1, 2, 3, 4, 5],
        emg_data=None,          # Should contain extended_emg & covariance_matrix
        batch_size=40,          # e.g., how many waveforms to handle at once
        forward_batch_size=100, # e.g., batch_size for probing phase
        extension_factor=10     # "R" factor for EMGProcessor.get_separation_vectors_torch
    ):
        """
        Constructor for ConditionOptimizer. We do not set the learnable parameters
        here, because we will do that inside optimize(...) to allow different
        initial_input each time if desired.
        """
        super().__init__()
        self.device = device
        self.margin = margin
        self.indices = indices
        self.biomime_muap_gen = biomime.to(self.device)

        self.batch_size = batch_size
        self.forward_batch_size = forward_batch_size
        self.extension_factor = extension_factor

        # Store references to the EMG training data / covariance
        # We expect something like: emg_data.tensors['extended_emg'], emg_data.tensors['covariance_matrix']
        self.emg_data = emg_data
        if self.emg_data is not None:
            self.training_data = emg_data.tensors["extended_emg"].to(self.device)
            self.cov_matrix = emg_data.tensors["covariance_matrix"].to(self.device)
        else:
            self.training_data = None
            self.cov_matrix = None

        # We will create self.opt_params later. For now, just a placeholder:
        self.opt_params = None
        self.biomime_input = None

    def forward(self):
        """
        Forward pass:
        1. Copies self.biomime_input
        2. Replaces specified indices with self.opt_params
        3. Calls biomime_muap_gen(...) to generate waveforms
        """
        if self.opt_params is None or self.biomime_input is None:
            raise ValueError("opt_params or biomime_input is None. Call optimize(...) or set initial parameters first.")

        # Make a copy so we don't mutate the original
        biomime_input = self.biomime_input.clone()
        biomime_input[:, self.indices] = self.opt_params

        # Forward through the BioMime generator
        # multiply by margin if your model expects [0, 1] input
        output = self.biomime_muap_gen(biomime_input * self.margin)
        return output

    def get_conditions(self):
        """
        Returns the current conditions (the learned parameters rescaled by margin).
        """
        if self.opt_params is None:
            return None
        return self.opt_params * self.margin

    # -----------------------------------------------------------------------
    # 1) Monte Carlo (LHS) sampling function
    # -----------------------------------------------------------------------
    def sample_monte_carlo(self, probe_points=10000, forward_batch_size=100):
        """
        1. Sample 'probe_points' in [0, 1]^6 using LHS.
        2. Evaluate them via the biomime_muap_gen to get waveforms.
        3. Compute negative kurtosis -> pick the best 'batch_size' points.
        4. Return the chosen 'best' initial conditions (shape [batch_size, 6]).
        """
        if self.emg_data is None:
            raise ValueError("emg_data was not provided. Required for optimization.")

        st = time()
        print(">>> Starting Monte Carlo (LHS) sampling...")
    
        probe_points = probe_points + (self.forward_batch_size - (probe_points % self.forward_batch_size))  # Ensure divisibility by batch_size

        # (A) Latin Hypercube Sampling in [0,1]^6
        lhs_sampler = qmc.LatinHypercube(d=6)
        lhs_points = qmc.scale(lhs_sampler.random(probe_points), 0, 1)  # shape: (probe_points, 6)
        lhs_tensor = torch.tensor(lhs_points, dtype=torch.float32, device='cpu')

        # (B) Evaluate in chunks on the GPU
        # biomime_muap_gen(...) can be expensive, so do it in batches of size self.batch_size
        torch.cuda.empty_cache()
        # forward batch size is different from the full GD batch size as no gradients are needed
        batched = lhs_tensor.view(-1, forward_batch_size, 6)
        all_muaps = []
        with torch.no_grad():
            for i in tqdm(range(batched.shape[0]), desc="Evaluating LHS batches"):
                batch_in = batched[i].to(self.device)
                batch_muaps = self.biomime_muap_gen(batch_in)  # shape e.g. [batch_size, 96, 320]
                all_muaps.append(batch_muaps.detach())

        # Combine all results
        muaps_lhs = torch.cat(all_muaps, dim=0)  # shape => [probe_points, 96, 320]
        # Possibly reshape for EMGProcessor usage
        muaps_lhs = muaps_lhs.reshape(probe_points, 96, 320).permute(0, 2, 1).float()  # [N, 320, 96]

        # (C) Compute negative kurtosis
        filters = EMGProcessor.get_separation_vectors_torch(muaps_lhs, R=self.extension_factor)
        source_est = filters.T @ self.cov_matrix @ self.training_data
        lhs_loss = -torch_kurtosis(source_est)  # shape [probe_points]
        
        # (D) Sort & pick best
        sorted_lhs = lhs_loss.sort()
        best_indices = sorted_lhs.indices[: self.batch_size]
        best_conds = lhs_tensor[best_indices.cpu()]  # shape => [batch_size, 6]

        print(f"<<< LHS sampling done. Elapsed: {time()-st:.2f} sec.")
    
        # save the forward information
        self.lhs_tensor = lhs_tensor
        self.lhs_loss = lhs_loss
        return best_conds  # shape [batch_size, 6]

    # -----------------------------------------------------------------------
    # 2) Gradient-based optimization function
    # -----------------------------------------------------------------------
    def optimize(self, initial_input, num_iterations=100, lr=1e-2):
        """
        Given an 'initial_input' of shape [batch_size, num_params],
        we (re-)initialize our learnable parameters (self.opt_params)
        and run gradient-based optimization for 'num_iterations'.
        """
        if self.emg_data is None:
            raise ValueError("emg_data was not provided. Required for optimization.")

        st = time()
        print(">>> Starting gradient-based optimization...")

        # (A) Initialize the Param
        # We'll store entire input as biomime_input, but only the 'indices' portion is learnable
        self.biomime_input = initial_input.clone().detach().to(self.device)
        with torch.no_grad():
            self.opt_params = nn.Parameter(
                (initial_input[:, self.indices] / self.margin).clone().to(self.device)
            )

        # Make sure the param is registered for gradient
        if not hasattr(self, "params"):
            self.params = []
        self.params = [self.opt_params]
        optimizer = optim.Adam(self.params, lr=lr)

        kurt_per_epoch = []

        for epoch in range(num_iterations):
            optimizer.zero_grad()

            # Forward to get waveforms
            muaps = self.forward()  # [batch_size, 96, 320] (example)
            muaps = muaps.reshape(self.batch_size, 96, 320).permute(0, 2, 1).float()

            # Separation vectors
            filters = EMGProcessor.get_separation_vectors_torch(muaps, R=self.extension_factor)
            source_est = filters.T @ self.cov_matrix @ self.training_data

            # Negative kurtosis => we sum
            kurt_per_mu = -torch_kurtosis(source_est)  # shape [batch_size]
            loss = kurt_per_mu.sum()

            loss.backward()
            optimizer.step()

            kurt_per_epoch.append(kurt_per_mu.detach().cpu().numpy())

            if (epoch + 1) % 10 == 0:
                avg_loss = loss.item() / self.batch_size
                print(f"Epoch [{epoch+1}/{num_iterations}] | Loss: {avg_loss:.6f} | Sum Kurt: {loss.item():.6f}")

        final_muaps = self.forward().detach().cpu().numpy()
        final_conditions = self.get_conditions().detach().cpu().numpy()

        print(f"<<< Optimization done. Elapsed: {time()-st:.2f} sec.")
        return final_muaps, final_conditions, kurt_per_epoch

    # -----------------------------------------------------------------------
    # 3) Full convenience method
    # -----------------------------------------------------------------------
    def full_optimize(self, probe_points=10000, num_iterations=100, lr=1e-2):
        """
        Runs the entire pipeline:
          1) sample_monte_carlo(...) to get best initial conditions
          2) optimize(...) with those conditions
        Returns final MUAPs, final conditions, kurtosis history
        """
        # (1) LHS / Monte Carlo to get best initial conditions
        best_conds = self.sample_monte_carlo(probe_points=probe_points, forward_batch_size=self.forward_batch_size)

        # (2) Gradient-based optimization using those best conditions
        final_muaps, final_conditions, kurt_history = self.optimize(
            initial_input=best_conds,
            num_iterations=num_iterations,
            lr=lr
        )

        return final_muaps, final_conditions, kurt_history
