import numbers
import torch
import torch.nn.functional as F
from typing import Union, Sequence, Optional
from scipy.ndimage._filters import gaussian_filter as scipy_gaussian_filter
import warnings

_MAX_REFLECT_BATCH = 2 ** 16 - 1  # CUDA reflect padding limit along batch dim


## ----------- Gaussian Filter ----------- ##

def _gaussian_kernel1d_torch(sigma, order, radius, device):
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = torch.arange(0, order + 1, device=device)
    sigma2 = sigma * sigma
    x = torch.arange(-radius, radius + 1, device=device)
    phi_x = torch.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        return phi_x
    else:
        q = torch.zeros(order + 1, device=device)
        q[0] = 1
        D = torch.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = torch.diag(torch.ones(order, device=device)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.matmul(q)
        q = (x[:, None] ** exponent_range).matmul(q)
        return q * phi_x

def _normalize_sequence_param(value, length, name, caster):
    if (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        if len(value) != length:
            raise ValueError(
                f"Parameter `{name}` must match number of axes ({length}); got length {len(value)}."
            )
        return [caster(v) for v in value]
    return [caster(value)] * length

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, numbers.Integral):
        axes = (int(axes),)
    normalized = []
    for ax in axes:
        ax_int = int(ax)
        if ax_int < 0:
            ax_int += ndim
        if ax_int < 0 or ax_int >= ndim:
            raise ValueError(f"Axis {ax} is out of bounds for tensor with {ndim} dims.")
        if ax_int in normalized:
            raise ValueError(f"Axis {ax_int} is specified more than once.")
        normalized.append(ax_int)
    return tuple(normalized)

def _reflect_pad_1d(x, pad_left, pad_right):
    if pad_left == 0 and pad_right == 0:
        return x
    length = x.shape[-1]
    if length == 1:
        # reflect mode is undefined here; replicate emulates constant reflection
        return _chunked_pad(x, (pad_left, pad_right), mode='replicate')
    padded = x
    left, right = pad_left, pad_right
    while left > 0 or right > 0:
        max_left = max(padded.shape[-1] - 1, 1)
        max_right = max(padded.shape[-1] - 1, 1)
        step_left = min(left, max_left)
        step_right = min(right, max_right)
        padded = _chunked_pad(padded, (step_left, step_right), mode='reflect')
        left -= step_left
        right -= step_right
    return padded

def _chunked_pad(tensor, pad_shape, mode='reflect'):
    batch = tensor.shape[0]
    if batch <= _MAX_REFLECT_BATCH or not tensor.is_cuda:
        return F.pad(tensor, pad_shape, mode=mode)
    chunks = []
    for start in range(0, batch, _MAX_REFLECT_BATCH):
        end = min(start + _MAX_REFLECT_BATCH, batch)
        chunks.append(F.pad(tensor[start:end], pad_shape, mode=mode))
    return torch.cat(chunks, dim=0)

def _pad_along_axis(tensor, axis, pad_left, pad_right):
    if pad_left == 0 and pad_right == 0:
        return tensor
    moved = tensor.movedim(axis, -1)
    orig_shape = moved.shape
    flat = moved.reshape(-1, 1, orig_shape[-1])
    padded = _reflect_pad_1d(flat, pad_left, pad_right)
    new_length = orig_shape[-1] + pad_left + pad_right
    padded = padded.reshape(*orig_shape[:-1], new_length)
    return padded.movedim(-1, axis)

def _apply_gaussian_filter1d(x, sigma, order, axis):
    axis = axis % x.ndim
    sd = float(sigma)
    if sd < 0:
        raise ValueError("Sigma must be non-negative.")
    if sd <= 1e-15:
        return x
    radius = int(4.0 * sd + 0.5)
    kernel = _gaussian_kernel1d_torch(sd, int(order), radius, x.device).to(x.dtype)
    # torch conv is correlation; flip weights for convolution parity with SciPy
    kernel = kernel.flip(0).view(1, 1, -1)
    pad = kernel.shape[-1] // 2
    moved = x.movedim(axis, -1)
    orig_shape = moved.shape
    length = orig_shape[-1]
    flat = moved.reshape(-1, 1, length)
    padded = _reflect_pad_1d(flat, pad, pad)
    filtered = F.conv1d(padded, kernel)
    filtered = filtered.reshape(orig_shape)
    return filtered.movedim(-1, axis)

def gaussian_filter_torch(
        input: torch.Tensor, 
        sigma: Union[float, Sequence[float]],
        order: Union[int, Sequence[int]] = 0,
        axes: Optional[Sequence[int]] = None):
    """
    Multidimensional Gaussian filter, mimiking scipy.ndimage.gaussian_filter but in PyTorch.
    If input is on CPU, reverts to using SciPy implementation.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be filtered.
    sigma : float or sequence of floats
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    order : int or sequence of ints, optional
        The order of the filter along each axis is given as a sequence
        of integers, or as a single number. An order of 0 corresponds
        to convolution with a Gaussian kernel. A positive order
        corresponds to convolution with that derivative of a Gaussian.
    axes : None or sequence of ints, optional
        Axes over which to compute the Gaussian filter. If None (default),
        the filter is computed over all axes.

    Returns
    -------
    filtered : torch.Tensor
        The filtered tensor.

    NOTE:
    - `mode` from gaussian_filter_scipy not implemented here; only 'reflect' mode is supported.
    - `cval`, `truncate`, and `radius` parameters are not implemented here. Should use default behaviors
    - if input is on cpu, reverts to using scipy implementation (faster on CPU)
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("`input` must be a torch.Tensor.")
    if input.ndim == 0:
        return input.clone()

    if input.get_device() == -1:
        return torch.as_tensor(scipy_gaussian_filter(input.numpy(), sigma=sigma, order=order, axes=axes))

    result = input if (input.dtype.is_floating_point or input.is_complex()) else input.to(torch.get_default_dtype())
    axes = _normalize_axes(axes, result.ndim)
    if len(axes) == 0:
        return result.clone()

    sigmas = _normalize_sequence_param(sigma, len(axes), "sigma", float)
    orders = _normalize_sequence_param(order, len(axes), "order", int)

    filtered = result
    modified = False
    for axis, s, o in zip(axes, sigmas, orders):
        if o < 0:
            raise ValueError("Order must be non-negative.")
        if s <= 1e-15:
            continue
        filtered = _apply_gaussian_filter1d(filtered, s, o, axis)
        modified = True
    return filtered if modified else filtered.clone()


## ----------- Median Filter ----------- ##
from scipy.ndimage import median_filter as scipy_median_filter

def median_filter_scipy(input, size=None, axes=None):
    """
    Wrapper for scipy.ndimage.median_filter for pytorch tensors.
    """
    inp_dev = input.device
    inp_dtype = input.dtype
    result = torch.as_tensor(scipy_median_filter(input.cpu().numpy(), size=size, axes=axes))
    return result.to(device=inp_dev, dtype=inp_dtype)

def _normalize_median_sizes(size, axes_len):
    base = 3 if size is None else size
    sizes = _normalize_sequence_param(base, axes_len, "size", int)
    for i, val in enumerate(sizes):
        if val <= 0:
            raise ValueError("Median filter `size` entries must be positive.")
        elif val % 2 == 0:
            # warn that need odd, so make odd
            warn_msg = "Median filter `size` entries should be odd; incremented by 1 to make odd."
            warnings.warn(warn_msg, UserWarning)
            sizes[i] = val + 1
    return sizes

def median_filter_torch(
        input: torch.Tensor,
        size: Optional[Union[int, Sequence[int]]] = 3,
        axes: Optional[Sequence[int]] = None):
    """
    Calculate a multidimensional median filter.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to be filtered.
    size : int or tuple of int, optional
        Size of the filter. If a single int is provided, the same size
        is used for all axes. If a tuple is provided, its length must
        match the number of axes to be filtered.
    axes : tuple of int or None, optional
        If None, `input` is filtered along all axes. Otherwise,
        `input` is filtered along the specified axes. When `axes` is
        specified, any tuples used for `size`, `origin`, and/or `mode`
        must match the length of `axes`. The ith entry in any of these tuples
        corresponds to the ith entry in `axes`.

    Returns
    -------
    median_filter : ndarray
        Filtered array. Has the same shape as `input`.

    NOTE:
    - only 'reflect' mode is supported.
    - CPU tensors fall back to `scipy.ndimage.median_filter` for parity.
    - `footprint`, `output`, `cval` and `origin` parameters are not implemented here. Should use default behaviors
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("`input` must be a torch.Tensor.")
    if input.ndim == 0:
        return input.clone()
    input_shape = input.shape

    axes = _normalize_axes(axes, input.ndim)
    if len(axes) == 0:
        return input.clone()

    if input.device.type == 'cpu':
        return median_filter_scipy(input, size=size, axes=axes)

    sizes = _normalize_median_sizes(size, len(axes))
    axes_sizes = sorted(zip(axes, sizes), key=lambda x: x[0])

    padded = input
    used = False
    for axis, sz in axes_sizes:
        if sz <= 1:
            continue
        pad_left = sz // 2
        pad_right = sz - pad_left - 1
        padded = _pad_along_axis(padded, axis, pad_left, pad_right)
        used = True

    if not used:
        return padded.clone()

    windows = padded
    windowed = False
    for axis, sz in axes_sizes:
        if sz <= 1:
            continue
        windows = windows.unfold(axis, sz, 1)
        windowed = True

    if not windowed:
        return input.clone()

    windows = windows.reshape(*input_shape, -1)
    median = windows.median(dim=-1).values
    return median.reshape(input.shape)
