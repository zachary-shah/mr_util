import torch
from typing import Optional

from .utils import hilbert
from .unwrap import torch_unwrap_1d

__all__ = [
    "instantaneous_frequency",
]

def instantaneous_frequency(
    x: torch.Tensor,
    dt: float,
    dim: int = -1,
    smooth: bool = False,
    smooth_window_len: int = 100,
    pad_width: Optional[int] = 500,
    clip_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute instantaneous frequency of a real-valued signal.

    Parameters
    ----------
    x: torch.Tensor
        The real-valued signal to compute the instantaneous frequency of.
        Shape (..., T, ...)
    dt: float
        The sampling interval in seconds.
    dim: int
        The dimension over which to compute the instantaneous frequency.
        Default is -1.
    smooth: bool
        Whether to apply smoothing to the instantaneous frequency.
    smooth_window_len: int
        The length of the smoothing window.
    pad_width: Optional[int]
        The number of points to pad on each side of the signal.
    clip_max: Optional[float]
        The maximum value of the instantaneous frequency.

    Returns
    -------
    f_inst: torch.Tensor
        The instantaneous frequency.
        Shape (..., T, ...)
    """
    if dim != -1:
        x = x.moveaxis(dim, -1)

    if pad_width is not None and pad_width > 0:
        # pad last dimension with reflection
        x = torch.nn.functional.pad(x, (pad_width, pad_width), mode='reflect')

    # phase of analytic signal
    x_a = hilbert(x, dim=-1).angle()
    
    # unwrap phase
    phase = torch_unwrap_1d(x_a, dim=-1)

    # time derivatie of phase
    f_inst = torch.diff(phase, append=phase[..., -1:], dim=-1) / dt / (2 * torch.pi)

    # smooth
    if smooth and smooth_window_len and (smooth_window_len > 0):
        if smooth_window_len % 2 == 0:
            smooth_window_len += 1
        kernel = torch.ones(smooth_window_len).to(f_inst.device).to(f_inst.dtype) / smooth_window_len
        batch_dims = f_inst.shape[:-1]
        f_inst = f_inst.reshape(-1, f_inst.shape[-1])
        f_inst = torch.nn.functional.conv1d(
            f_inst[:, None, :],
            kernel[None, None, :], 
            stride=1, 
            padding=smooth_window_len // 2,
        )[:, 0]
        f_inst = f_inst.reshape(*batch_dims, -1)

    if pad_width is not None and pad_width > 0:
        f_inst = f_inst[..., pad_width:-pad_width]

    f_inst = f_inst.clip(min=0, max=clip_max)

    if dim != -1:
        f_inst = f_inst.moveaxis(-1, dim)

    return f_inst
