import sys

import random

import torch
import torch.fft as fft

import numpy as np
from numbers import Real
from sklearn.utils import check_random_state

from typing import Any
import torch.nn as nn


class RandomDropLeads(nn.Module):
    """
        Randomly drop leads from the signal.
    """
    def __init__(self, probability=0.5):
        super(RandomDropLeads, self).__init__()
        self.probability = probability

    def forward(self, signal):
        if self.training:
            leads_to_remove = np.random.random(signal.shape[-1]) < self.probability
            leads_to_remove[1] = False  # never remove lead II
            signal[..., leads_to_remove] = 0
        return signal
    
class Jitter(object):
    """
        Add gaussian noise to the sample.
    """
    def __init__(self, sigma=0.2, amplitude=0.6, prob=1.0) -> None:
        self.sigma = sigma
        self.amplitude = amplitude
        self.prob = prob

    def __call__(self, sample) -> Any:
        # 0. If the probability is 0, return the original sample.
        if self.prob == 0: return sample
        
        # 1. Generate a mask for applying jitter based on probability. This creates a boolean tensor where True indicates jitter should be applied.
        mask = (torch.rand(sample.shape[0], device=sample.device) < self.prob).float()  # [batch_size]

        # 2. Expand the mask to match the tensor dimensions. We need to broadcast the mask along all dimensions *except* the batch dimension.
        mask = mask.view(mask.size(0), *([1] * (sample.ndim - 1)))  # [batch_size, 1, 1, 1, ...]

        # 3. Generate noise for the *entire* batch.
        noise = torch.randn_like(sample) * self.sigma

        # 4. Calculate the amplitude scaling for the *entire* batch.
        amplitude_scaling = self.amplitude * sample

        # 5. Apply the jitter only where the mask is True. This is done using element-wise multiplication and addition.
        jittered_tensor = sample + mask * amplitude_scaling * noise

        return jittered_tensor

    
class FTSurrogate(object):
    """
    FT surrogate augmentation of a single EEG channel, as proposed in [1]_.
    Code (modified) from https://github.com/braindecode/braindecode/blob/master/braindecode/augmentation/functional.py 
    

    Parameters
    ----------
    X : torch.Tensor
        EEG input example.
    phase_noise_magnitude: float
        Float between 0 and 1 setting the range over which the phase
        pertubation is uniformly sampled:
        [0, `phase_noise_magnitude` * 2 * `pi`].
    channel_indep : bool
        Whether to sample phase perturbations independently for each channel or
        not. It is advised to set it to False when spatial information is
        important for the task, like in BCI.
    random_state: int | numpy.random.Generator, optional
        Used to draw the phase perturbation. Defaults to None.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.

    References
    ----------
    .. [1] Schwabedal, J. T., Snyder, J. C., Cakmak, A., Nemati, S., &
       Clifford, G. D. (2018). Addressing Class Imbalance in Classification
       Problems of Noisy Signals by using Fourier Transform Surrogates. arXiv
       preprint arXiv:1806.08675.
    """
    def __init__(self, phase_noise_magnitude, channel_indep=False, seed=None, prob=1.0) -> None:
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.seed = seed
        self.prob = prob
        self._new_random_fft_phase = {
            0: self._new_random_fft_phase_even,
            1: self._new_random_fft_phase_odd
        }

    def _new_random_fft_phase_odd(self, c, n, device='cpu', seed=None):
        rng = check_random_state(seed)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((c, (n - 1) // 2))
        ).to(device)

        return torch.cat([
            torch.zeros((c, 1), device=device),
            random_phase,
            -torch.flip(random_phase, [-1]).to(device=device)
        ], dim=-1)
    
    def _new_random_fft_phase_even(self, c, n, device='cpu', seed=None):
        rng = check_random_state(seed)
        random_phase = torch.from_numpy(
            2j * np.pi * rng.random((c, n // 2 - 1))
        ).to(device)

        return torch.cat([
            torch.zeros((c, 1), device=device),
            random_phase,
            torch.zeros((c, 1), device=device),
            -torch.flip(random_phase, [-1]).to(device=device)
        ], dim=-1)

    def __call__(self, sample) -> Any:
        if self.prob == 0: return sample    

        mask = torch.rand(sample.shape[0], device=sample.device) < self.prob  # [batch_size]

        if isinstance(self.phase_noise_magnitude, torch.Tensor):
            # Ensure magnitude tensor is on the correct device and has the right shape
            self.phase_noise_magnitude = self.phase_noise_magnitude.to(sample.device)
            if self.phase_noise_magnitude.ndim == 1:
                 # Expand to [batch_size, 1, 1, ...] for broadcasting.
                magnitude = self.phase_noise_magnitude.view(sample.shape[0], *([1] * (sample.ndim - 1)))
            else:
                raise ValueError("phase_noise_magnitude tensor must be 1-dimensional (batch_size) or a scalar.")

        elif isinstance(self.phase_noise_magnitude, Real):
             # Expand to [batch_size, 1, 1, ...] for broadcasting.
            magnitude = torch.full((sample.shape[0], *([1] * (sample.ndim - 1))), self.phase_noise_magnitude, device=sample.device, dtype=sample.dtype)  #Correctly handles the broadcasting of a scalar magnitude.
        else:
           raise TypeError("phase_noise_magnitude must be a float or a 1D torch.Tensor")
        
        mask = mask.view(mask.size(0), *([1] * (sample.ndim - 1))).float()  # [batch_size, 1, 1, ...]

        # 4. Compute FFT of the entire batch.
        f = fft.fft(sample.double(), dim=-1)  # [batch_size, ..., num_samples]
        n = f.shape[-1]

        # 5. Generate random phases for the *entire batch*.
        num_channels = f.shape[-2] if self.channel_indep else 1
        random_phase = self._new_random_fft_phase[n % 2](
            sample.shape[0] * num_channels, n, device=sample.device, seed=self.seed
        )  # [batch_size * num_channels, n]

        # 6. Reshape and repeat random_phase if necessary.
        if self.channel_indep:
            random_phase = random_phase.view(sample.shape[0], num_channels, n) # [batch_size, num_channels, n]
            random_phase = random_phase.reshape(sample.shape[0], *f.shape[1:-1], n)
        else:
            # Repeat the same phase for all channels within each sample.
            random_phase = random_phase.view(sample.shape[0], 1, n)  # [batch_size, 1, n]
            random_phase = random_phase.expand(sample.shape[0], *f.shape[1:-1], n) # [batch_size, num_channels, ..., n]

        # 7. Apply phase shift and inverse FFT.
        f_shifted = f * torch.exp(magnitude * random_phase * 1j) # Use complex number multiplication
        shifted = fft.ifft(f_shifted, dim=-1)
        sample_transformed = shifted.real.float()

        # 8. Apply the transformation selectively based on the mask.
        return mask * sample_transformed + (1 - mask) * sample


    
class Rescaling(object):
    """
        Randomly rescale features of the sample.
    """
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma

    def __call__(self, sample) -> Any:
        sample = sample * torch.normal(mean=torch.Tensor([1]), std=torch.Tensor([self.sigma]))
        return sample


class Shift(object):
    """
        Randomly shift the signal in the time domain.
    """
    def __init__(self, fs=360, padding_len_sec=1) -> None:
        self.padding_len = fs * padding_len_sec # padding len in ticks 

    def __call__(self, sample) -> Any:
        # define padding size 
        left_pad = int(torch.rand(1) * self.padding_len)
        right_pad = self.padding_len - left_pad

        # zero-pad the sample
        # note: the signal length is now extended by self.padding_len
        padded_sample = torch.nn.functional.pad(sample, (left_pad, right_pad), value=0)

        # get back to the original signal length
        if torch.rand(1) < 0.5:
            return padded_sample[..., :sample.shape[-1]]
        else:
            return padded_sample[..., right_pad:sample.shape[-1]+right_pad]

class CropResizing(object):
    """
        Randomly crop the sample and resize to the original length.
    """
    def __init__(self, lower_bnd=0.8, upper_bnd=0.8, fixed_crop_len=None, start_idx=None, resize=False, fixed_resize_len=None) -> None:
        self.lower_bnd = lower_bnd
        self.upper_bnd = upper_bnd
        self.fixed_crop_len = fixed_crop_len
        self.start_idx = start_idx
        self.resize = resize
        self.fixed_resize_len = fixed_resize_len

    def __call__(self, sample) -> Any:
        sample_dims = sample.dim()
        
        # define crop size
        if self.fixed_crop_len is not None:
            crop_len = self.fixed_crop_len
        else:
            # randomly sample the target length from a uniform distribution
            crop_len = int(sample.shape[0]*np.random.uniform(low=self.lower_bnd, high=self.upper_bnd))
        
        # define cut-off point
        if self.start_idx is not None:
            start_idx = self.start_idx
        else:
            # randomly sample the starting point for the cropping (cut-off)
            try:
                start_idx = np.random.randint(low=0, high=sample.shape[0]-crop_len)
            except ValueError:
                # if sample.shape[-1]-crop_len == 0, np.random.randint() throws an error
                start_idx = 0

        # crop and resize the signal
        if self.resize == True:
            # define length after resize operation
            if self.fixed_resize_len is not None:
                resize_len = self.fixed_resize_len
            else:
                resize_len = sample.shape[-1]

            # crop and resize the signal
            cropped_sample = torch.zeros_like(sample[..., :resize_len])
            if sample_dims == 2:
                for ch in range(sample.shape[-2]):
                    resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[ch, start_idx:start_idx+crop_len])
                    cropped_sample[ch, :] = torch.from_numpy(resized_signal)
            elif sample_dims == 3:
                for f_bin in range(sample.shape[-3]):
                    for ch in range(sample.shape[-2]):
                        resized_signal = np.interp(np.linspace(0, crop_len, num=resize_len), np.arange(crop_len), sample[f_bin, ch, start_idx:start_idx+crop_len])
                        cropped_sample[f_bin, ch, :] = torch.from_numpy(resized_signal)
            else:
                sys.exit('Error. Sample dimension does not match.')
        else:
            # only crop the signal
            cropped_sample = torch.zeros_like(sample)
            cropped_sample = sample[start_idx:start_idx+crop_len, ...]

        return cropped_sample

class Interpolation(object):
    """
        Undersample the signal and interpolate to initial length.
    """
    def __init__(self, step=2, prob=1.0) -> None:
        self.step = step
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            sample_sub = sample[..., ::self.step]
            sample_interpolated = np.ones_like(sample)
            
            sample_dims = sample.dim()
            if sample_dims == 2:
                for ch in range(sample.shape[-2]):
                    sample_interpolated[ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[ch])
            elif sample_dims == 3:
                for f_bin in range(sample.shape[-3]):
                    for ch in range(sample.shape[-2]):
                        sample_interpolated[f_bin, ch] = np.interp(np.arange(0, sample.shape[-1]), np.arange(0, sample.shape[-1], step=self.step), sample_sub[f_bin, ch])
            else:
                sys.exit('Error. Sample dimension does not match.')

            return torch.from_numpy(sample_interpolated)
        else:
            return sample

class Masking(object):
    """
        Randomly zero-mask the sample.
        Got this from https://stackoverflow.com/questions/70092136/how-do-i-create-a-random-mask-matrix-where-we-mask-a-contiguous-length
        Don't touch the code!
    """
    def __init__(self, factor:float=0.75, fs:int=360, patch_size_sec:float=1, masking_mode="random", prob=1.0) -> None:
        self.factor = factor                    # fraction to be masked out
        self.patch_size = int(patch_size_sec * fs)   # patch_size[ticks] = patch_size[sec] * fs[Hz]
        self.masking_mode = masking_mode
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            # create the mask
            mask = torch.ones_like(sample)

            # determine the number of patches to be masked
            nb_patches = round(self.factor * sample.shape[-1] / self.patch_size)
            
            indices_weights = np.random.random((mask.shape[0], nb_patches + 1))

            number_of_ones = mask.shape[-1] - self.patch_size * nb_patches

            ones_sizes = np.round(indices_weights[:, :nb_patches].T
                                * (number_of_ones / np.sum(indices_weights, axis=-1))).T.astype(np.int32)
            ones_sizes[:, 1:] += self.patch_size

            zeros_start_indices = np.cumsum(ones_sizes, axis=-1)

            if self.masking_mode == "block":
                for zeros_idx in zeros_start_indices[0]:
                    mask[..., zeros_idx: zeros_idx + self.patch_size] = 0
            else:
                for sample_idx in range(len(mask)):
                    for zeros_idx in zeros_start_indices[sample_idx]:
                        mask[sample_idx, zeros_idx: zeros_idx + self.patch_size] = 0

            return sample * mask
        else:
            return sample
    
    
class FrequencyShift(object):
    """
    Adds a shift in the frequency domain to all channels.
    Note that here, the shift is the same for all channels of a single example.
    Code (modified) from https://github.com/braindecode/braindecode/blob/master/braindecode/augmentation/functional.py

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.
    delta_freq : float
        The amplitude of the frequency shift (in Hz).
    sfreq : float
        Sampling frequency of the signals to be transformed.
    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    def __init__(self, delta_freq=0, s_freq=360, prob=1.0) -> None:
        self.delta_freq = delta_freq
        self.s_freq = s_freq
        self.prob = prob

    def _analytic_transform(self, X):
        if torch.is_complex(X):
            raise ValueError("X must be real.")

        N = X.shape[-1]
        f = fft.fft(X, N, dim=-1)
        h = torch.zeros_like(f)
        if N % 2 == 0:
            h[..., 0] = h[..., N // 2] = 1
            h[..., 1:N // 2] = 2
        else:
            h[..., 0] = 1
            h[..., 1:(N + 1) // 2] = 2

        return fft.ifft(f * h, dim=-1)

    def _nextpow2(self, n):
        """Return the first integer N such that 2**N >= abs(n)."""

        return int(np.ceil(np.log2(np.abs(n))))

    def _frequency_shift(self, X, fs, f_shift):
        """
        Shift the specified signal by the specified frequency.
        See https://gist.github.com/lebedov/4428122
        """
        nb_channels, N_orig = X.shape[-2:]

        # Pad the signal with zeros to prevent the FFT invoked by the transform
        # from slowing down the computation:
        N_padded = 2 ** self._nextpow2(N_orig)
        t = torch.arange(N_padded, device=X.device) / fs
        padded = torch.nn.functional.pad(X, (0, N_padded - N_orig))

        analytical = self._analytic_transform(padded)
        if isinstance(f_shift, (float, int, np.ndarray, list)):
            f_shift = torch.as_tensor(f_shift).float()

        reshaped_f_shift = f_shift.repeat(N_padded, nb_channels).T
        shifted = analytical * torch.exp(2j * np.pi * reshaped_f_shift * t)

        return shifted[..., :N_orig].real.float()

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            sample_transformed = self._frequency_shift(
                X=sample,
                fs=self.s_freq,
                f_shift=self.delta_freq,
            )

            return sample_transformed
        else:
            return sample
    
class TimeFlip(object):
    """
        Flip the signal vertically.
    """
    def __init__(self, prob=1.0) -> None:
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            return torch.flip(sample, dims=[-1])
        else:
            return sample
    
class SignFlip(object):
    """
        Flip the signal horizontally.
    """
    def __init__(self, prob=1.0) -> None:
        self.prob = prob

    def __call__(self, sample) -> Any:
        if np.random.uniform() < self.prob:
            return -1*sample
        else:
            return sample
        
class SpecAugment(object):
    """
        Randomly masking frequency or time bins of signal's short-time Fourier transform.
        See https://arxiv.org/pdf/2005.13249.pdf
    """
    def __init__(self, masking_ratio=0.2, n_fft=120) -> None:
        self.masking_ratio = masking_ratio
        self.n_fft = n_fft

    def __call__(self, sample) -> Any:
        sample_dim = sample.dim()

        if sample_dim < 3:
            masked_sample = self._mask_spectrogram(sample)
        elif sample_dim == 3:
            # perform masking separately for all entries in the first dimension 
            # and eventually concatenate the masked entries to retrieve the intial shape 
            masked_sample = torch.Tensor()
            for i in range(sample.shape[0]):
                masked_sub_sample = self._mask_spectrogram(sample[i])
                masked_sample = torch.cat((masked_sample, masked_sub_sample.unsqueeze(0)), dim=0)
        else: 
            print(f"Augmentation was not built for {sample_dim}-D input")

        return masked_sample

    def _mask_spectrogram(self, sample):
        sample_length = sample.shape[-1]

        # compute the Fourier transform
        spec = torch.stft(sample, n_fft=self.n_fft, return_complex=True)

        if random.random() < 0.5:
            # frequency domain
            masked_block_size = int(spec.shape[-2]*self.masking_ratio)
            start_idx = random.randint(0, spec.shape[-2] - masked_block_size)
            end_idx = start_idx + masked_block_size

            # mask the bins
            spec[..., start_idx:end_idx, :] = 0.+0.j
        else:
            # time domain
            masked_block_size = int(spec.shape[-1]*self.masking_ratio)
            start_idx = random.randint(0, spec.shape[-1] - masked_block_size)
            end_idx = start_idx + masked_block_size

            # mask the bins
            spec[..., start_idx:end_idx] = 0.+0.j

        # perform the inverse Fourier transform to get the new signal
        masked_sample = torch.istft(spec, n_fft=self.n_fft, length=sample_length)

        return masked_sample