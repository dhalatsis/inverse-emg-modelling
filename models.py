import torch
import torch.nn as nn
import torch.optim as optim


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


class ConditionOptimizer(nn.Module):
    def __init__(
        self, 
        biomime, 
        initial_input, 
        zi, 
        indices=[0,1,2,3,4,5], 
        device='cuda:0'
    ):
        super(ConditionOptimizer, self).__init__()
        self.device = device
        self.biomime = biomime.to(device)  # Pretrained BioMime network

        self.indices = indices

        # Make the portion of input that is learnable an nn.Parameter
        # We use logit of the initial_input so we can pass it through a sigmoid
        self.opt_params = nn.Parameter(
            torch.logit(initial_input.clone().detach()[:, indices])
        )

        # The rest of the input, which remains fixed, also store in self
        # but *not* as a parameter
        self.biomime_input = torch.logit(initial_input.clone().detach().to(device))

        # Some additional parameter or condition (zi)
        self.zi = zi.to(device)
        self.zi.requires_grad_(False)  # not learnable

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
        output = self.biomime.sample(
            n_mus, 
            self.sig(biomime_input), 
            self.device, 
            self.zi.repeat(n_mus, 1)
        )
        return output

    def get_conditions(self):
        """
        Return the current conditions (i.e., the parameters in [0, 1]).
        This can be used to check the learned values after optimization.
        """
        return self.sig(self.opt_params)


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