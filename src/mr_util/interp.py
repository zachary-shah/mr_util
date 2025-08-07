from typing import Literal, Optional

import torch
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
