from typing import Literal, Optional

import torch
import numpy as np
from scipy.interpolate import interp1d as scipy_interp1d

from .ext.torchcubicspline import torch_cubic_interpolate
from .unwrap import torch_unwrap_1d

__all__ = [
    "interp1d_complex",
]

TORCH_INTERP_METHODS = ["cubic"]
SCIPY_INTERP_KINDS = ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
INTERP_METHODS = Literal["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]


def __cubic_interp_torch(
    y: torch.Tensor,
    x: torch.Tensor,
    x_new: torch.Tensor,
    channel_dim: int = 1,
    **kwargs,
) -> torch.Tensor:
    """
    Wrapper for patrick kidger's implementation of cubic interpolation
    """
    assert channel_dim > 0
    y = y.moveaxis(channel_dim, -1)
    y = y.moveaxis(0, -2)
    y = torch_cubic_interpolate(x, y, x_new)
    y = y.moveaxis(-2, 0)
    y = y.moveaxis(-1, channel_dim)
    return y


def __interp_scipy(
    y: torch.Tensor, x: torch.Tensor, x_new: torch.Tensor, kind: str = "cubic", **kwargs
) -> torch.Tensor:
    device = x.device
    interp_func = scipy_interp1d(
        x.cpu().numpy(), y.cpu().numpy(), kind=kind, axis=0, **kwargs
    )
    return torch.from_numpy(interp_func(x_new.cpu().numpy())).to(device).to(y.dtype)


def interp1d_complex(
    y: torch.Tensor,
    x: torch.Tensor,
    x_new: torch.Tensor,
    dim: int = 0,
    method: INTERP_METHODS = "cubic",
    backend: Optional[Literal["scipy", "torch"]] = None,
    mag_phase: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Interpolate a complex signal along a single dimension.

    Parameters
    ----------
    y : torch.Tensor
        Tensor of arbitrary shape, with interpolation dimension along dim of length N.
    x : torch.Tensor
        Sampled indices of y, shape (N,)
    x_new :
        New indices to interpolate y at, shape (M,)
    dim : int, optional
        Dimension along which to interpolate, by default 0.
    method : INTERP_METHODS, optional
        Interpolation method to use, by default 'cubic'.
        The power of this wrapper is using cubic interpolation on complex data with torch backend.
        Torch cubic interpolation is 10x faster than scipy but has 10x interpolation error (still negligible in performance for most applications).
    backend : Optional[Literal['scipy', 'torch']], optional
        Backend to use for interpolation, by default None.
        If None, will use 'torch' if available, otherwise 'scipy'.
    mag_phase : bool, optional
        If true, interpolates magnitude/phase of signal rather than on real/imaginary.
    kwargs : dict, optional
        Additional keyword arguments to pass to the interpolation function.

    Returns
    -------
    torch.Tensor
        Interpolated tensor with the same shape as y, but with the interpolation dimension replaced by
        length M of x_new.

    Notes
    -----
    - `torch` interpolation is 10x faster than `scipy` for cubic interpolation, but has 10x interpolation error.
    - linear, quadratic, and zero-th order interpolations not yet implemented for `torch` backend, so must use `scipy`.

    TODO:
    - support for other interpolation methods in torch backend.
    - add kb interpolation
    """

    assert (
        method in INTERP_METHODS.__args__
    ), f"Interpolation method {method} not supported. Choose from {INTERP_METHODS.__args__}."

    assert backend in [
        "scipy",
        "torch",
        None,
    ], f"Interpolation backend {backend} not supported. Choose from 'scipy' or 'torch'."

    # Defaults
    if backend is None:
        backend = "torch" if method in TORCH_INTERP_METHODS else "scipy"

    if backend == "torch":
        assert (
            method in TORCH_INTERP_METHODS
        ), f"Interpolation method {method} not supported for torch backend. Choose from {TORCH_INTERP_METHODS}."
    elif backend == "scipy":
        assert (
            method in SCIPY_INTERP_KINDS
        ), f"Interpolation method {method} not supported for scipy backend. Choose from {SCIPY_INTERP_KINDS}."

    if dim != 0:
        y = y.moveaxis(dim, 0)

    assert (
        y.shape[0] == x.shape[0]
    ), f"Length of time vector x ({x.shape[0]}) must match first dimension of y ({y.shape[0]})."

    if torch.is_complex(y) and mag_phase:
        mag = torch.abs(y)
        phs = torch.angle(y)

        phs = torch_unwrap_1d(phs, dim=0)

        if backend == "torch" and method == "cubic":
            mag_interp = __cubic_interp_torch(mag, x, x_new, **kwargs)
            phs_interp = __cubic_interp_torch(phs, x, x_new, **kwargs)
        else:
            mag_interp = __interp_scipy(mag, x, x_new, kind=method, **kwargs)
            phs_interp = __interp_scipy(phs, x, x_new, kind=method, **kwargs)

        y_new = mag_interp * torch.exp(1j * phs_interp)
    else:
        if backend == "torch" and method == "cubic":
            y_new = __cubic_interp_torch(y, x, x_new, **kwargs)
        else:
            y_new = __interp_scipy(y, x, x_new, kind=method, **kwargs)

    if dim != 0:
        y_new = y_new.moveaxis(0, dim)

    return y_new


class interp1d_no_extrap:

    def __init__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor,
                 kind: str = 'linear', 
                 axis: int = -1,
                 copy: bool = True, 
                 bounds_error: bool = None, 
                 fill_value: float = np.nan,
                 assume_sorted: bool = False,
                 extrapolate_nearest_neighbor: bool = True,
                 ):
        """
        Interpolation wrapper around scipy's interp1d, with optional nearest-neighbor extrapolation.

        Parameters:
            x (torch.Tensor): The input tensor containing the x-coordinates of the data points.
            y (torch.Tensor): The input tensor containing the y-coordinates of the data points.
            kind (str, optional): Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', etc.). Default is 'linear'.
            axis (int, optional): Axis along which to interpolate. Default is -1.
            copy (bool, optional): If True, the input arrays are copied. Default is True.
            bounds_error (bool or None, optional): If True, an error is raised when interpolation is attempted outside the range of x. If False, out-of-bounds values are assigned fill_value. Default is None.
            fill_value (float or str, optional): Value to use for points outside the interpolation range. Default is torch.nan. If extrapolate_nearest_neighbor is True, this is set to 'extrapolate'.
            assume_sorted (bool, optional): If True, x is assumed to be sorted. Default is False.
            extrapolate_nearest_neighbor (bool, optional): If True, extrapolated values are replaced with the nearest neighbor value instead of standard extrapolation. If False, behaves like scipy's interp1d. Default is True.
        Notes:
            - When extrapolate_nearest_neighbor is True, bounds_error must be None.
            - Converts input tensors to numpy arrays for compatibility with scipy's interp1d.
            - Stores lower and upper bounds and corresponding y-values for nearest-neighbor extrapolation.
        """

        x = x.cpu().numpy()
        y = y.cpu().numpy()

        if extrapolate_nearest_neighbor:
            assert bounds_error is None, "Cannot set bounds_error with extrapolate_nearest_neighbor=True"
            fill_value = 'extrapolate'
            
        interp_func = scipy_interp1d(x, y, kind=kind, axis=axis, copy=copy, bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)

        self.ndim = y.ndim
        self.kind = kind
        self.axis = axis
        self.interp_func = interp_func
        self.extrapolate_nearest_neighbor = extrapolate_nearest_neighbor

        # Find min and max x vals
        self.lower_bound = np.min(x)
        self.upper_bound = np.max(x)

        self.lower_loc = np.argmin(x)
        self.upper_loc = np.argmax(x)

        # save lower and upper vals
        lower_slc = [slice(None) for _ in range(self.ndim)]
        lower_slc[axis] = self.lower_loc
        self.lower_val = y[tuple(lower_slc)]

        upper_slc = [slice(None) for _ in range(self.ndim)]
        upper_slc[axis] = self.upper_loc
        self.upper_val = y[tuple(upper_slc)]
        
    def __call__(self, xnew: torch.Tensor) -> torch.Tensor:
        dev = xnew.device
        dtype = xnew.dtype

        # base interpolation call
        ynew = self.interp_func(xnew.cpu().numpy())

        ynew = torch.tensor(ynew, device=dev, dtype=dtype)

        if self.extrapolate_nearest_neighbor:
            # extrapolate nearest neighbor
            lower_extrap_inds = torch.where(xnew < self.lower_bound)[0]
            upper_extrap_inds = torch.where(xnew > self.upper_bound)[0]

            for l in lower_extrap_inds:
                lower_slc = [slice(None) for _ in range(self.ndim)]
                lower_slc[self.axis] = l
                ynew[tuple(lower_slc)] = self.lower_val

            for u in upper_extrap_inds:
                upper_slc = [slice(None) for _ in range(self.ndim)]
                upper_slc[self.axis] = u
                ynew[tuple(upper_slc)] = self.upper_val

        return ynew