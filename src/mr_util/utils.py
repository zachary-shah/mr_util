from math import sqrt
from typing import Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import windows
from tqdm import tqdm

from .filter import gaussian_filter_torch, median_filter_torch
from .spatial import (
    TRITON_AVAILABLE,
    SPATIAL_RESIZE_METHODS,
    RESIZE_METHODS,
    WINDOW_METHODS,
    DEFAULT_WINDOW,
    _cubic_spline_prefilter_kernel,
    _fourier_resize,
    _spatial_resize,
    _spatial_resize_poly_lagrange,
    _spatial_resize_poly_cubic,
    _cubic_spline_filter_axis,
    hilbert,
    nd_windowed_filter,
    rotation_matrix_3d,
    rotation_matrix,
    gen_grd,
)

__all__ = [
    "batch_iterator",
    "tqdm_batch_iterator",
    "resize",
    "fft",
    "ifft",
    "hilbert",
    "nd_windowed_filter",
    "rotation_matrix_3d",
    "rotation_matrix",
    "gen_grd",
    "spatial_resize",
    "spatial_filter",
    "normalize",
    "binary_dilation_1d",
]

def batch_iterator(total: int, batch_size: int, return_len: bool = False):
    """
    Get iteratable list of indices for batched iteration

    Parameters
    ----------
    total : int
        Total number of elements to iterate over
    batch_size : int
        Batch size

    Returns
    -------
    batch_list : list[tuple]
        List of tuples of the form (start, end) where start is the
        starting index of the batch and end is the ending index of the batch.
    """
    assert total > 0, f"batch_iterator called with {total} elements"
    delim = list(range(0, total, batch_size)) + [total]
    iterator = zip(delim[:-1], delim[1:])
    if return_len:
        N = len(delim) - 1
        return iterator, N
    return iterator


def tqdm_batch_iterator(total: int, batch_size: Optional[int] = None, **tqdm_kwargs):
    """Convenience function for wrapping batch_iterator with a progress bar"""
    if batch_size is None:
        tqdm_total = 1
    else:
        tqdm_total = -(-total // batch_size)
    return tqdm(
        batch_iterator(total, batch_size),
        total=tqdm_total,
        **tqdm_kwargs,
    )


def resize(input: torch.Tensor, oshape: Tuple[int, ...], padval=None) -> torch.Tensor:
    """
    Resize with zero-padding or cropping.

    Parameters
    ----------
        input (torch.tensor): Input array.
        oshape (tuple of ints): Output shape.

    Returns
    -------
        torch.tensor: Zero-padded or cropped result.
    """

    assert len(input.shape) == len(
        oshape
    ), "Input and output must have same number of dimensions."

    ishape = input.shape

    if ishape == oshape:
        return input

    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape, oshape)]

    copy_shape = [
        min(i - si, o - so) for i, si, o, so in zip(ishape, ishift, oshape, oshift)
    ]

    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    if torch.is_tensor(input):
        output = torch.zeros(oshape, dtype=input.dtype, device=input.device)
    else:
        output = np.zeros(oshape, dtype=input.dtype)
    if padval is not None:
        output[:] = padval

    output[oslice] = input[islice]

    return output


def fft(
    x: torch.Tensor, o_im_shape: Tuple[int, ...], center: bool = True
) -> torch.Tensor:
    """
    Compute the cartesian FFT of image data with shape (..., im_shape)
    with k-space centering support

    Parameters
    ----------
    x : torch.Tensor
        Input image data with shape (..., in_img_shape)
    o_im_shape : tuple
        Desired output image shape
    center : bool, optional
        Whether to center the k-space, by default True

    Returns
    -------
    x : torch.Tensor
        Output k-space data with shape (..., o_im_shape)
    """

    fftdims = tuple(range(-len(o_im_shape), 0))

    newshape = (*x.shape[: -len(o_im_shape)], *o_im_shape)

    if center:
        x = resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm="ortho")
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.fftn(x, s=o_im_shape, dim=fftdims, norm="ortho")

    return x


def ifft(
    x: torch.Tensor, o_im_shape: Tuple[int, ...], center: bool = True
) -> torch.Tensor:
    """
    Compute the inverse cartesian FFT of k-space data with shape (..., im_shape)
    with k-space centering support

    Parameters
    ----------
    x : torch.Tensor
        Input k-space data with shape (..., in_img_shape)
    o_im_shape : tuple
        Desired output image shape
    center : bool, optional
        Whether to center the k-space, by default True

    Returns
    -------
    x : torch.Tensor
        Output image data with shape (..., o_im_shape)
    """

    fftdims = tuple(range(-len(o_im_shape), 0))

    newshape = (*x.shape[: -len(o_im_shape)], *o_im_shape)

    if center:
        x = resize(x, newshape)
        x = torch.fft.ifftshift(x, dim=fftdims)
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm="ortho")
        x = torch.fft.fftshift(x, dim=fftdims)
    else:
        x = torch.fft.ifftn(x, s=o_im_shape, dim=fftdims, norm="ortho")

    return x


def spatial_resize(
    x: torch.Tensor,
    im_size: tuple,
    method: RESIZE_METHODS = "poly",
    window: Optional[WINDOW_METHODS] = None,
    order: int = 3,
    mag_phase: bool = False,
) -> torch.Tensor:
    """
    Resize a spatial tensor to a new spatial size.

    Parameters:
    -----------
    x : (torch.Tensor)
        The input tensor with shape (..., *inp_im_size)
    im_size : (tuple)
        The size of the image to resize to
    method : (Optional[str])
        The method to use for resizing, options are:
            - 'bilinear': linear interpolation (default)
            - 'bicubic': cubic interpolation
            - 'nearest': nearest sample interpolation
            - 'fourier': fourier interpolation
            - 'poly': polynomial interpolation
    window : (Optional[str])
        The window to apply to the fourier interpolation
    order : (Optional[int])
        The order of the polynomial interpolation,
        if method = poly.
    mag_phase : (bool)
        Whether to resize the magnitude and phase separately,
        if complex. Default is real/imag.

    Returns:
    --------
    x : (torch.Tensor)
        The resized tensor with shape (..., *im_size)
    """

    inp_im_size = x.shape[-len(im_size) :]
    if inp_im_size == im_size:
        return x

    is_np = False
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        is_np = True

    is_bool = False
    if x.dtype == torch.bool:
        x = x.to(torch.float32)
        is_bool = True

    # reshape to (B, *im_size)
    is_batched = True
    if len(im_size) == x.ndim:
        x = x[
            None,
        ]
        is_batched = False
    else:
        orig_batch = x.shape[: -len(im_size)]
        orig_im_size = x.shape[-len(im_size) :]
        x = x.reshape((-1, *orig_im_size))

    if method == "fourier":
        assert not mag_phase, "Fourier resize does not support mag_phase"
        x = _fourier_resize(x, im_size, window)
    elif method == "poly":
        if order == 3 and TRITON_AVAILABLE:
            resize_poly = lambda value: _spatial_resize_poly_cubic(value, im_size)
        else:
            resize_poly = lambda value: _spatial_resize_poly_lagrange(
                value, im_size, order=order
            )
        if mag_phase:
            xabs = resize_poly(x.abs())
            xarg = resize_poly(x.angle())
            x = xabs * torch.exp(1j * xarg)
        else:
            x = resize_poly(x)
    elif method in SPATIAL_RESIZE_METHODS.__args__:
        if mag_phase:
            xabs = _spatial_resize(x.abs(), im_size, method=method)
            xarg = _spatial_resize(x.angle(), im_size, method=method)
            x = xabs * torch.exp(1j * xarg)
        else:
            x = _spatial_resize(x, im_size, method=method)
    else:
        raise ValueError(
            f"Unknown resize method: {method}. Must be one of {RESIZE_METHODS}."
        )

    if is_batched:
        x = x.reshape((*orig_batch, *im_size))
    else:
        x = x[0]

    if is_bool:
        x = x.to(torch.bool)

    if is_np:
        x = x.cpu().numpy()

    return x


def spatial_filter_scipy_backend(
    x: torch.Tensor,
    im_size: Tuple[int, ...],
    gaussian_filter_size: Optional[Union[float, Tuple[float, ...]]] = None,
    median_filter_size: Optional[Union[int, Tuple[int, ...]]] = None,
) -> torch.Tensor:
    """
    Apply spatial filters to image.
    DEPRECIATED: uses scipy backends.

    Parameters:
    -----------
    x : torch.Tensor
        The input tensor with shape (..., *im_size)
    im_size : Tuple[int, ...]
        The size of the image to filter
    gaussian_filter_size : Optional[Tuple[int, ...]]
        The size of the gaussian filter to apply, if None, no gaussian filter is applied
    median_filter_size : Optional[Tuple[int, ...]]
        The size of the median filter to apply, if None, no median filter is applied

    Returns:
    --------
    x : torch.Tensor
        The filtered tensor with shape (..., *im_size)
    """

    if gaussian_filter_size is None and median_filter_size is None:
        return x

    torch_info = None
    if isinstance(x, torch.Tensor):
        torch_info = (x.device, x.dtype)
        x = x.cpu().numpy()

    # dims to filter
    filt_dims = tuple(range(-len(im_size), 0))

    if median_filter_size is not None:
        if isinstance(median_filter_size, (int, float)):
            if isinstance(median_filter_size, float):
                median_filter_size = int(median_filter_size)
            median_filter_size = (median_filter_size,) * len(im_size)
        elif len(median_filter_size) != len(im_size):
            raise ValueError(
                "median_filter_size must be an int or a tuple of the same length as im_size."
            )

        # filter real/imag seperately if complex
        is_complex = np.iscomplexobj(x)
        if is_complex:
            x = np.stack([x.real, x.imag], axis=0)
        x = median_filter(x, size=median_filter_size, axes=filt_dims)
        if is_complex:
            x = x[0] + 1j * x[1]

    if gaussian_filter_size is not None:
        if isinstance(gaussian_filter_size, (int, float)):
            gaussian_filter_size = (gaussian_filter_size,) * len(im_size)
        elif len(gaussian_filter_size) != len(im_size):
            raise ValueError(
                "gaussian_filter_size must be an int or a tuple of the same length as im_size."
            )
        x = gaussian_filter(x, sigma=gaussian_filter_size, axes=filt_dims)

    if torch_info is not None:
        x = torch.from_numpy(x).to(torch_info[0]).to(torch_info[1])

    return x


def spatial_filter(
    x: torch.Tensor,
    im_size: Tuple[int, ...],
    gaussian_filter_size: Optional[Union[float, Tuple[float, ...]]] = None,
    median_filter_size: Optional[Union[int, Tuple[int, ...]]] = None,
    filt_dims: Optional[Tuple[int, ...]] = None,
    use_depreciated: bool = False,
) -> torch.Tensor:
    """
    Apply spatial filters to (last dims) of image.

    Parameters:
    -----------
    x : torch.Tensor
        The input tensor with shape (..., *im_size)
    im_size : Tuple[int, ...]
        The size of the image to filter
    gaussian_filter_size : Optional[Tuple[int, ...]]
        The size of the gaussian filter to apply, if None, no gaussian filter is applied
    median_filter_size : Optional[Tuple[int, ...]]
        The size of the median filter to apply, if None, no median filter is applied
    filt_dims: Optional[Tuple[int, ...]]
        The dimensions to filter, if None, uses last dims

    Returns:
    --------
    x : torch.Tensor
        The filtered tensor with shape (..., *im_size)
    """

    if use_depreciated:
        return spatial_filter_scipy_backend(
            x, im_size, gaussian_filter_size, median_filter_size
        )

    if gaussian_filter_size is None and median_filter_size is None:
        return x

    # dims to filter
    if filt_dims is None:
        filt_dims = tuple(range(-len(im_size), 0))

    if median_filter_size is not None:
        # filter real/imag seperately if complex
        is_complex = torch.is_complex(x)    
        if is_complex:
            x = torch.stack([x.real, x.imag], dim=0)
        x = median_filter_torch(x, size=median_filter_size, axes=filt_dims)
        if is_complex:
            x = x[0] + 1j * x[1]

    if gaussian_filter_size is not None:
        x = gaussian_filter_torch(x, sigma=gaussian_filter_size, axes=filt_dims)

    return x


def normalize(shifted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Normalize the shifted tensor to match the target tensor scale.
    """
    scale = torch.linalg.vecdot(
        shifted.flatten().abs(),
        target.flatten().abs(),
        dim=0,
    ) / (torch.linalg.vector_norm(shifted.flatten(), dim=0) ** 2 + 1e-8)
    shifted = shifted * scale
    return shifted


def binary_dilation_1d(
    x: torch.Tensor,
    iterations: int = 1,
    dim: int = -1,
) -> torch.Tensor:
    """
    Perform 1D binary dilation on a tensor.

    Parameters:
    -----------
    x : torch.Tensor
        The input tensor with shape (..., *im_size)
    iterations : int
        The number of iterations of dilation to apply.
    dim : int
        The dimension over which to perform the dilation.
        Default is -1.

    Returns:
    --------
    x : torch.Tensor
        The dilated tensor with shape (..., *im_size)
    """
    x = x.bool()
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if iterations == 0:
        return x

    dim = dim % x.ndim
    x_move = x.movedim(dim, -1)
    x_flat = x_move.reshape(-1, 1, x_move.shape[-1]).to(torch.float32)

    for _ in range(iterations):
        x_flat = torch.nn.functional.max_pool1d(x_flat, kernel_size=3, stride=1, padding=1)

    return x_flat.bool().reshape(x_move.shape).movedim(-1, dim)
