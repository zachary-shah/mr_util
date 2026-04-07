
import torch
from typing import Optional, List, Tuple
from scipy.signal import decimate

from .utils import (
    spatial_filter, 
    tqdm_batch_iterator, 
    binary_dilation_1d,
)
from .ifreq import instantaneous_frequency

__all__ = [
    "rf_spike_filter",
]


def bool_array_to_str(arr, N=50):
    # print arr as a string of . for 0 and | for 1
    # downsample array to N
    arr = decimate(arr, arr.shape[0] // N).round().astype(int)
    return ''.join(['-' if x else '_' for x in arr])


def rf_spike_filter(
    ksp: torch.Tensor,
    trj: torch.Tensor,
    grad: torch.Tensor,
    dt: float,
    device: Optional[torch.device] = None,
    # Filtering Parameters
    kr_exp: float = 1.8,
    height_factor: float = 2.0,
    detrend_smooth_len: int = 500,
    peak_width: int = 5,
    freq_ranges: Optional[List[Tuple[float, float]]] = None,
    return_valid = False,
    freq_trim_end: bool = True,
    batch_size: int = 256,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Filter out spikes in the k-space data.
    Does this by normalizing data with heuristics, searching for peaks with some peak_width,

    Parameters
    ----------
    ksp: torch.Tensor
        The k-space data to filter.
        Shape (..., T,S), for T the readout dimension, S the phase encode / shot dimension
    trj: torch.Tensor
        The trajectory in cycles/image.
        Shape (..., T, S, d)
    grad: torch.Tensor
        The gradient in G/cm.
        Shape (T, S, d)
    dt: float
        Sampling interval in seconds.
    kr_exp: float
        Exponent of the k-space radius to normalize the data by.
    height_factor: float
        Factor of 99th percentile which classifies a spike
    detrend_smooth_len: int
        Length of the smoothing window for de-trending.
    peak_width: int
        Max width to filter around each peak detected.
    freq_ranges: List[Tuple[float, float]]
        List of frequency ranges in the gradient where spikes are expected.
        Will only keep peaks detected in these ranges.
    return_valid: bool
        By default, returns mask of invalid indices.
        If True, returns mask of valid indices.
    batch_size: int
        Batch size for processing.
    verbose: bool
        Whether to print verbose output.

    Returns
    -------
    peak_mask: torch.Tensor
        Mask of invalid indices. Shape (..., T, S)
        If return_valid is True, returns mask of valid indices.
    """
    if device is None:
        device = ksp.device
    T, S = ksp.shape[-2:]

    # First, normalize data by kr
    ksp = ksp * (torch.norm(trj, dim=-1).to(ksp.device) ** kr_exp)

    # rest of computations should batch
    batch_dims = ksp.shape[:-2]
    ksp = ksp.reshape(-1, T, S)
    B = ksp.shape[0]
    batch_size = max(batch_size // S, 1)
    
    peak_mask = torch.zeros_like(ksp, dtype=torch.bool)
    for iL, iR in tqdm_batch_iterator(B, batch_size, disable=not verbose):
        ksp_batch = ksp[iL:iR].abs()
        
        # high-pass filter the data
        ksp_trend = spatial_filter(
            ksp_batch, (T,), 
            gaussian_filter_size=detrend_smooth_len, 
            median_filter_size=((detrend_smooth_len // 2) * 2 + 1,),
            filt_dims=(-2,), 
        )
        ksp_batch = (ksp_batch - ksp_trend).abs()

        # detect peaks as locations above spike height threshold
        spike_scale = (ksp_batch.quantile(0.99, dim=1, keepdim=True))
        peak_mask_batch = (ksp_batch >= spike_scale * height_factor)

        # potentially dilate the mask to cover wider regions, but only to above some threshold
        if peak_width > 1:
            expand_mask_batch = ksp_batch >= ksp_batch.quantile(0.9, dim=1, keepdim=True)
            peak_mask_batch = binary_dilation_1d(peak_mask_batch, iterations=peak_width - 1, dim=1)
            peak_mask_batch = peak_mask_batch & expand_mask_batch

        peak_mask[iL:iR] = peak_mask_batch

    # frequency mask
    if freq_ranges is not None:
        pad_width = min(grad.shape[0] // 20, 500)
        ifreqs = instantaneous_frequency(
            grad.to(device), dt, dim=0, 
            smooth=True,
            smooth_window_len=100,
            pad_width=pad_width,
            clip_max=5000, # max [Hz]
        ).mean(dim=(1, 2)) # (T,)
        ifreqs[:pad_width] = 0
        if freq_trim_end:
            ifreqs[-pad_width:] = 0

        freq_mask = torch.zeros_like(ifreqs, dtype=torch.bool)
        for fr in freq_ranges:
            mask = (ifreqs > fr[0]) & (ifreqs < fr[1])
            freq_mask = freq_mask | mask

        if verbose:
            print(f"RF spike detection range given gradient fedges={freq_ranges}:")
            print(bool_array_to_str(freq_mask.cpu().numpy()))

        peak_mask = peak_mask & freq_mask[None, :, None]

    return_slc = peak_mask.reshape(*batch_dims, T, S)

    if return_valid:
        return_slc = ~return_slc

    return return_slc
