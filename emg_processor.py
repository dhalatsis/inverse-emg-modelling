import numpy as np
import torch
from tqdm import tqdm
import scipy

class EMGDataset:
    def __init__(self, file_path):
        """
        Load dataset from an .npz file, including zi (Gaussian-related variable).
        """
        if not file_path.endswith(".npz"):
            raise ValueError("Only .npz file format is supported.")
        
        data = np.load(file_path)
        self.data_dict = {key: data[key] for key in data.files}
        self.tensors = {}

    def to_torch(self):
        """
        Convert all loaded numpy arrays into PyTorch tensors.
        """
        self.tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in self.data_dict.items()}

    def get_tensor(self, name):
        """
        Retrieve a specific tensor by name.
        """
        if name in self.tensors:
            return self.tensors[name]
        else:
            raise ValueError(f"Tensor '{name}' not found. Ensure you have called to_torch() first.")

    def add_noise(self, noise_std=0.4):
        """
        Add Gaussian noise to the EMG data and update the extended EMG and covariance matrix.

        Args:
            noise_std (float): Standard deviation of the Gaussian noise.
        """
        self.data_dict["emg"] += np.random.randn(*self.data_dict["emg"].shape) * noise_std
        R = self.data_dict['extended_emg'].shape[1] - self.data_dict['emg'].shape[1] + 1

        self.data_dict['extended_emg'] = EMGProcessor.extend_emg(self.data_dict['emg'], R)
        self.data_dict['covariance_matrix'] = EMGProcessor.get_inv_cov(self.data_dict['extended_emg'], explained_var=1-1e-14)

    def to_device(self, device='cpu'):
        """
        Move tensors to GPU/CPU.
        """
        self.tensors = {key: tensor.to(device) for key, tensor in self.tensors.items()}



class EMGProcessor:
    @staticmethod
    def generate_spike_trains(mu_count, duration, fs, Tmean=0.1, Tstd=0.03):
        """
        Generate motor unit spike trains.

        Args:
            mu_count (int): Number of motor units.
            duration (float): Duration of the signal in seconds.
            fs (float): Sampling frequency in Hz.
            Tmean (float): Mean inter-spike interval in seconds.
            Tstd (float): Standard deviation of inter-spike interval.

        Returns:
            np.ndarray: Spike trains of shape (mu_count, duration * fs).
            list: Discharge times for each motor unit.
        """
        spts = np.zeros((mu_count, int(fs * duration)))
        dts = []
        for mu_idx in range(mu_count):
            times = np.random.normal(loc=Tmean, scale=Tstd, size=int(duration / Tmean))
            times = np.cumsum(times)
            dts.append(times)
            times = (fs * times[times <= duration]).astype(int)
            spts[mu_idx, times] = 1
        return spts, dts

    @staticmethod
    def generate_emg(spts, muaps):
        """
        Generate EMG based on spike trains and simulated MUAPs.

        Args:
            spts (np.ndarray): Spike trains of shape (mu_count, duration * fs).
            muaps (np.ndarray): MUAPs of shape (mu_count, channels, window_length).

        Returns:
            np.ndarray: Generated EMG of shape (channels, duration * fs).
        """
        EMG = np.zeros((muaps.shape[1], spts.shape[1]))
        for mdx in tqdm(range(muaps.shape[0]), desc="Generating EMG"):
            for edx in range(muaps.shape[1]):
                EMG[edx, :] += np.convolve(spts[mdx, :], muaps[mdx, edx, :], mode='same')
        return EMG

    @staticmethod
    def generate_emg_causal(spts, muaps, device='cpu'):
        ''' Generate EMG based on spike trains and simulated MUAPs using torch. '''
        # Convert inputs to torch tensors
        muaps = torch.tensor(muaps, dtype=torch.float64)
        spts = torch.tensor(spts, dtype=torch.float64)
        
        # Reshape spike trains for convolution input
        spts = spts.view(1, spts.shape[0], 1, spts.shape[1])
        
        # Pad spike trains for causal convolution
        padding_total = muaps.shape[-1] - 1
        spts = torch.nn.functional.pad(spts, (padding_total, 0), mode='constant', value=0)
        
        # Prepare MUAPs as convolution kernels
        muaps = torch.transpose(muaps, dim0=0, dim1=1).unsqueeze(2)
        muaps = torch.flip(muaps, dims=[3])
        
        # Perform convolution to generate EMG
        EMG = torch.nn.functional.conv2d(spts.to(device), muaps.to(device))
        EMG = EMG.view(muaps.shape[0], EMG.shape[-1])
        
        return EMG.detach().numpy()
    
    @staticmethod
    def extend_emg(signal, ext_factor):
        """
        Extend EMG signals for given temporal and spatial shifts.

        Args:
            signal (np.ndarray): Original EMG signal of shape (channels, observations).
            ext_factor (int): Extension factor.

        Returns:
            np.ndarray: Extended EMG of shape (channels * ext_factor, observations + ext_factor - 1).
        """
        nchans, nobvs = signal.shape
        extended_template = np.zeros((nchans * ext_factor, nobvs + ext_factor - 1))
        for i in tqdm(range(ext_factor), desc="Extending EMG"):
            extended_template[nchans * i:nchans * (i + 1), i:nobvs + i] = signal
        return extended_template

    @staticmethod
    def whiten_emg(signal, explained_var=0.99):
        """
        Whiten the EMG signal to impose a covariance matrix equal to the identity matrix,
        keeping only the components that contribute to the specified explained variance.

        Args:
            signal (np.ndarray): EMG signal of shape (channels, observations).
            explained_var (float): Fraction of variance to retain.

        Returns:
            tuple: Whitened EMG, whitening matrix, and de-whitening matrix.
        """
        cov_mat = np.cov(signal, bias=True)
        evalues, evectors = scipy.linalg.eigh(cov_mat)
        sorted_idxs = np.argsort(evalues)[::-1]
        evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
        cum_explained_var = np.cumsum(evalues) / np.sum(evalues)
        valid_idxs = cum_explained_var <= explained_var
        evalues, evectors = evalues[valid_idxs], evectors[:, valid_idxs]
        diag_mat = np.diag(evalues)
        whitening_mat = evectors @ np.linalg.inv(np.sqrt(diag_mat)) @ evectors.T
        dewhitening_mat = evectors @ np.sqrt(diag_mat) @ evectors.T
        whitened_emg = whitening_mat @ signal
        return whitened_emg.real, whitening_mat, dewhitening_mat

    @staticmethod
    def get_separation_vectors_torch(muaps, R=None):
        """
        Generate separation vectors based on MUAPs.

        Args:
            muaps (torch.Tensor): MUAPs of shape (mu_count, channels, window_length).
            R (int, optional): Extension factor. Defaults to window_length.

        Returns:
            torch.Tensor: Separation vectors.
        """
        N, Nch, L = muaps.shape
        if R is None:
            R = L
        B = torch.zeros((Nch * R, N), device=muaps.device)
        for mdx in range(N):
            for l in range(R):
                B[l * Nch:(l + 1) * Nch, mdx] = muaps[mdx, :, R + 20 - l].ravel()
        return B
    
    @staticmethod
    def get_separation_vectors(muaps, R=None):
        """
        Generate separation vectors based on MUAPs.

        Args:
            muaps (torch.Tensor): MUAPs of shape (mu_count, channels, window_length).
            R (int, optional): Extension factor. Defaults to window_length.

        Returns:
            torch.Tensor: Separation vectors.
        """
        N, Nch, L = muaps.shape
        if R is None:
            R = L
        B = np.zeros((Nch * R, N))
        for mdx in range(N):
            for l in range(R):
                B[l * Nch:(l + 1) * Nch, mdx] = muaps[mdx, :, R + 20 - l].ravel()
        return B

    @staticmethod
    def get_inv_cov(signal, explained_var=0.99):
        """
        Get inverse covariance matrix with eigenvalue truncation for regularization.

        Args:
            signal (np.ndarray): Extended EMG signal.
            explained_var (float): Fraction of variance to retain.

        Returns:
            np.ndarray: Inverse covariance matrix.
        """
        cov_mat = np.cov(signal, bias=True)
        evalues, evectors = scipy.linalg.eigh(cov_mat)
        sorted_idxs = np.argsort(evalues)[::-1]
        evalues, evectors = evalues[sorted_idxs], evectors[:, sorted_idxs]
        cum_explained_var = np.cumsum(evalues) / np.sum(evalues)
        valid_idxs = cum_explained_var <= explained_var
        evalues, evectors = evalues[valid_idxs], evectors[:, valid_idxs]
        inv_cov = evectors @ np.diag(1 / evalues) @ evectors.T
        return inv_cov
