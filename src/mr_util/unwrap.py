import torch
from typing import Optional, Union
import numpy as np

from scipy import ndimage
from skimage.restoration import unwrap_phase as skimage_unwrap_phase

__all__ = [
    "nearest_neighbor_extrapolation",
    "unwrap_phase",
    "torch_unwrap_spatial",
    "torch_unwrap_1d",
]

def nearest_neighbor_extrapolation(p: np.ndarray,
                                   valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extrapolate NaN values in a 2D array using the nearest neighbor.
    
    Parameters
    ----------
    p : np.ndarray
        2D array with NaN values for extrapolation.
    valid_mask : np.ndarray, optional
        Boolean mask indicating valid (True) and invalid (False) values in the array.
        If None, the function will assume all values are valid.
    
    Returns
    ---------
    np.ndarray
        Array with NaN values extrapolated.
    """
    
    if valid_mask is None:
        valid_mask = ~np.isnan(p)
    
    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    
    return p[tuple(indices)]


def unwrap_phase(p: Union[np.ndarray, torch.Tensor],
                 isocenter: Optional[tuple] = None,
                 wrap_around: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Use skimage's unwrap_phase to unwrap a phase map.
    Assume that isocenter does not have a wrap (DC term does not wrap for SH basis).

    Parameters
    ----------
    p : np.ndarray or torch.Tensor
        Input phase map to unwrap. Should be 2D or 3D.
    isocenter : tuple, optional
        (x,y) coordinate indices of the isocenter. Default is the center of the phase map.
    wrap_around : bool, optional

    Returns
    -------
    np.ndarray or torch.Tensor
        Unwrapped phase map. If input is torch.Tensor, output will also be a torch.Tensor
    """
    # skimage only supports numpy
    device = None
    if isinstance(p, torch.Tensor):
        device = p.device
        p = p.detach().cpu().numpy()

    # potentially extrapolate nan values
    if any(np.isnan(p.flatten())):
        p = nearest_neighbor_extrapolation(p)

    if isocenter is None:
        isocenter = tuple([axs//2 for axs in p.shape])
    
    p_unwrapped = skimage_unwrap_phase(p, wrap_around=wrap_around)

    # remove 2pi bulk offsets
    p_center = p_unwrapped[isocenter]
    k = p_center + np.pi - np.mod(p_center + np.pi, 2*np.pi)
    p_unwrapped -= k   

    if device is not None:
        p_unwrapped = torch.tensor(p_unwrapped, dtype=torch.float32).to(device)

    return p_unwrapped 


class _UnwrapSpatial(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor,
                extrap_mask: torch.Tensor = None,
                isocenter: tuple = None,
                wrap_around: bool = False,) -> torch.Tensor:

        # prep input for phase unwrap operation
        input_dtype = x.dtype
        input_device = x.device
        input_requires_grad = x.requires_grad
        x_np = x.detach().cpu().numpy()

        # operation (batched unwrap and extrapolation on phase map)
        if extrap_mask is not None:
            extrap_mask_np = extrap_mask.detach().cpu().numpy()
            x_np = nearest_neighbor_extrapolation(x_np, extrap_mask_np)

        x_unwrapped = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            x_unwrapped[i] = unwrap_phase(
                x_np[i], 
                isocenter=isocenter, 
                wrap_around=wrap_around,
            )

        # restore to torch
        out = torch.tensor(
            x_unwrapped, 
            device=input_device, 
            dtype=input_dtype, 
            requires_grad=input_requires_grad
        )

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):

        grad_x = grad_mask = grad_isocenter = grad_wrap_around = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        
        return grad_x, grad_mask, grad_isocenter, grad_wrap_around


def torch_unwrap_spatial(p: torch.Tensor,
                         ndim: int,
                         period: Optional[float] = None,
                         valid_mask: torch.Tensor = None,
                         isocenter: tuple = None,
                         wrap_around: bool = False,) -> torch.Tensor:
    """
    Wrapper for 2D/3D spatial unwrapping of a phase map in Torch, with torch.autograd support.

    Parameters
    ----------
    p : torch.Tensor
        Input phase maps to unwrap of shape (..., *spatial)
    ndim : int
        Number of spatial dimensions (2 or 3).
    period : float, optional
        Wrap-around period. By default is 2π.
    valid_mask : torch.Tensor
        Mask to use for unwrapping, shape (..., *spatial). or (*spatial).
        Should be 1/True for valid data, 0/False for invalid data
    isocenter : tuple
        Isocenter for setting relative point for unwrap. If none, unwraps relative to center.
    wrap_around : bool
        If true, will unwrap assuming circular continuity across image

    Returns
    -------
    torch.Tensor
        Unwrapped phase map of shape (B, ...)
    """

    assert ndim in [2, 3], "ndim must be 2 or 3 for spatial unwrapping"

    batch_dims = p.shape[:-ndim]
    im_size = p.shape[-ndim:]
    if len(batch_dims) == 0:
        p = p.unsqueeze(0)
    p = p.reshape((-1, *im_size))

    if valid_mask is not None:
        valid_shape = valid_mask.shape[-ndim:]
        assert valid_shape == im_size, f"Expected shape {im_size} for valid_mask, but got {valid_shape}"
        # match batching
        if valid_mask.shape == (*batch_dims, *im_size):
            valid_mask = valid_mask.reshape((-1, *im_size))
        elif valid_mask.shape == im_size:
            valid_mask = valid_mask.unsqueeze(0).expand(p.shape[0], *im_size)

    if period is not None:  
        p = p / period * (2 * torch.pi)

    p = _UnwrapSpatial.apply(p, valid_mask, isocenter, wrap_around)

    if period is not None:
        p = p / (2 * torch.pi) * period
    
    if len(batch_dims) == 0:
        p = p.squeeze(0)
    else:
        p = p.reshape((*batch_dims, *im_size))
    
    return p


def torch_unwrap_1d(
    p: torch.Tensor,
    period: float = 2 * torch.pi,
    discont: Optional[float] = None,
    dim: int = -1,
) -> torch.Tensor:
    """
    Perform a 1D unwrapping operation on a tensor along a given dimension

    Parameters
    ----------
    p : torch.Tensor
        Tensor, to unwrap along dim `dim`.
    discont : float, optional
        Maximum discontinuity to consider when unwrapping (default period/2).
    Period : float, optional
        Wrap-around period (default 2π).
    dim : int, optional
        Dimension along which to unwrap (default last axis).
    
    Returns
    -------
    torch.Tensor
        Unwrapped tensor, same shape as p.
    """

    if discont is None:
        discont = period / 2

    p = p.clone() if p.requires_grad else p.detach().clone()

    dd = torch.diff(p, dim=dim)

    if p.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        interval_high = period // 2
        boundary_ambiguous = (period % 2 == 0)
    else:
        interval_high = period / 2
        boundary_ambiguous = True
    
    interval_low = -interval_high

    # map differences into the interval [low, low+period)
    ddmod = torch.remainder(dd - interval_low, period) + interval_low

    # fix the ambiguous boundary case so that +period/2 keeps its sign
    if boundary_ambiguous:
        mask = (ddmod == interval_low) & (dd > 0)
        ddmod[mask] = interval_high

    ph_correct = ddmod - dd

    # Tolerance check for discontinuities
    ph_correct = torch.where(
        torch.abs(dd) < discont,
        torch.zeros_like(ph_correct),
        ph_correct
    )

    # build output by cumulatively summing corrections
    up = p.clone().to(dd.dtype)
    sl = [slice(None)] * p.ndim
    sl[dim] = slice(1, None)
    csum = ph_correct.cumsum(dim=dim)
    up[tuple(sl)] = p[tuple(sl)] + csum

    return up
