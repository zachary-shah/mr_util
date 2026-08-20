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
    _spatial_resize,
    _spatial_resize_poly_lagrange,
    _spatial_resize_poly_cubic,
    _cubic_spline_filter_axis,
    hilbert,
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
    x: torch.Tensor, 
    o_im_shape: Optional[Tuple[int, ...]] = None, 
    dim: Optional[Union[int, Tuple[int, ...]]] = None, 
    center: bool = True, 
    norm = "ortho",
) -> torch.Tensor:
    """
    Compute the cartesian FFT of image data with shape (..., im_shape)
    with k-space centering support

    Parameters
    ----------
    x : torch.Tensor
        Input image data with shape (..., in_img_shape)
    o_im_shape : tuple
        Desired output image shape along FFT dimensions
    dim : tuple, optional
        The dimensions to perform the FFT on, if None, uses last dims
    center : bool, optional
        Whether to center the k-space, by default True

    Returns
    -------
    x : torch.Tensor
        Output k-space data with shape (..., o_im_shape)
    """

    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(o_im_shape, int):
        o_im_shape = (o_im_shape,)

    if dim is None and o_im_shape is not None:
        # assume last dims
        dim = tuple(range(-len(o_im_shape), 0))
    
    if o_im_shape is not None:
        # resize as needed
        newshape = list(tuple(x.shape))
        for d in dim:
            newshape[d] = o_im_shape[d]
        x = resize(x, tuple(newshape))
    
    if center:
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.fftn(x, dim=dim, norm=norm)
        x = torch.fft.fftshift(x, dim=dim)
    else:
        x = torch.fft.fftn(x, dim=dim, norm=norm)


    return x


def ifft(
    x: torch.Tensor, 
    o_im_shape: Optional[Tuple[int, ...]] = None, 
    dim: Optional[Union[int, Tuple[int, ...]]] = None, 
    center: bool = True, 
    norm = "ortho",
) -> torch.Tensor:
    """
    Compute the inverse cartesian FFT of k-space data with shape (..., im_shape)
    with k-space centering support

    Parameters
    ----------
    x : torch.Tensor
        Input k-space data with shape (..., in_img_shape)
    o_im_shape : tuple
        Desired output image shape along FFT dimensions
    dim : tuple, optional
        The dimensions to perform the FFT on, if None, uses last dims
    center : bool, optional
        Whether to center the k-space, by default True

    Returns
    -------
    x : torch.Tensor
        Output image data with shape (..., o_im_shape)
    """

    if isinstance(dim, int):
        dim = (dim,)
    if isinstance(o_im_shape, int):
        o_im_shape = (o_im_shape,)

    if dim is None and o_im_shape is not None:
        # assume last dims
        dim = tuple(range(-len(o_im_shape), 0))
    
    if o_im_shape is not None:
        # resize as needed
        newshape = list(tuple(x.shape))
        for d in dim:
            newshape[d] = o_im_shape[d]
        x = resize(x, tuple(newshape))

    if center:
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.ifftn(x, dim=dim, norm=norm)
        x = torch.fft.fftshift(x, dim=dim)
    else:
        x = torch.fft.ifftn(x, dim=dim, norm=norm)

    return x


def nd_windowed_filter(
    w_shape: Tuple[int, ...],
    window: WINDOW_METHODS = "hann",
    oshape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Get n-dimensional windowed fourier filter for data with shape im_shape, where the window is defined by w_shape.

    Returns window of shape w_shape, or a window padded to oshape if provided.
    """
    if isinstance(w_shape, int):
        w_shape = (w_shape,)
    if isinstance(oshape, int):
        oshape = (oshape,)

    if oshape is None:
        oshape = w_shape
    else:
        assert len(oshape) == len(
            w_shape
        ), "Output shape must have same number of dimensions as window shape."
        assert all(
            [oshape[i] >= w_shape[i] for i in range(len(oshape))]
        ), "Output shape must be larger than window shape."

    d = len(w_shape)
    assert d <= 3, "Only 1, 2, or 3 dimensions supported."

    if window == "hamming":
        wfunc = lambda x: windows.hamming(x)
    elif window == "kaiser":
        wfunc = lambda x: windows.kaiser(x, beta=14)
    elif window == "gaussian":
        wfunc = lambda x: windows.gaussian(x, std=sqrt(max(w_shape)))
    elif window == "tukey":
        wfunc = lambda x: windows.tukey(x, alpha=0.5)
    elif window == "hann":
        wfunc = lambda x: windows.hann(x)
    elif window == "boxcar":
        wfunc = lambda x: windows.boxcar(x)
    else:
        raise ValueError(
            f"Unknown window type: {window}. Must be one of {WINDOW_METHODS.__args__}."
        )

    # form n-d window
    if d == 1:
        out = wfunc(w_shape[0])
    elif d == 2:
        out = wfunc(w_shape[0])[:, None] @ wfunc(w_shape[1])[None, :]
    elif d == 3:
        out_2d = wfunc(w_shape[0])[:, None] @ wfunc(w_shape[1])[None, :]
        out = out_2d[:, :, None] * wfunc(w_shape[2])[None, None, :]

    # zero pad window to oshape
    if oshape != w_shape:
        out = resize(out, oshape)

    return out


def _fourier_resize(
    x: torch.Tensor, new_shape: Tuple[int, ...], window: Optional[WINDOW_METHODS] = None
) -> torch.Tensor:
    """
    Take an image and reshape it to new_shape using fourier padding

    Parameters:
    -----------
    x : torch.Tensor
        The input tensor with shape (..., *inp_im_size)
    new_shape : Tuple[int, ...]
        The size of the image to resize to
    window : Optional[WINDOW_METHODS]
        The window to apply to the fourier interpolation. Options:
            - 'boxcar': boxcar window (default)
            - 'hamming': hamming window
            - 'hann': hann window
            - 'blackman': blackman window
            - 'kaiser': kaiser window

    Returns:
    --------
    x : torch.Tensor
        The resized tensor with shape (..., *new_shape)
    """

    isComplex = False

    if window is None:
        window = DEFAULT_WINDOW

    if torch.is_complex(x):
        isComplex = True

    ndim = len(new_shape)
    fft_shape = x.shape[-ndim:]
    abs_max = x.abs().max()

    x = fft(x, fft_shape)

    if window != "boxcar":
        wind = (
            torch.from_numpy(nd_windowed_filter(fft_shape, window=window))
            .to(x.device)
            .to(x.dtype)
        )
        for _ in range(x.ndim - wind.ndim):
            wind = wind[None, ...]
        x = x * wind

    x = ifft(x, new_shape)

    if not isComplex:
        x = x.real

    x = x / x.abs().max() * abs_max

    return x


def spatial_resize(
    x: torch.Tensor,
    im_size: tuple,
    method: RESIZE_METHODS = "poly",
    window: Optional[WINDOW_METHODS] = None,
    order: int = 3,
    mag_phase: bool = False,
    preserve_dc_position: bool = True,
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
    preserve_dc_position : (bool)
        Only supported for method="poly". Default False keeps the existing
        align-corners behavior (pins array endpoints 0<->0, N-1<->N-1), which
        only coincides with the N//2 DC/FFT-center convention used elsewhere
        in this codebase (gen_grd, fft, ifft, resize) when both the input and
        output sizes are odd -- any resize involving an even size introduces a
        sub-pixel offset between the resized array's center and that convention.
        If True, the resize instead maps output index im_size[i]//2 to input
        index inp_im_size[i]//2 exactly, matching the N//2 convention, at the
        cost of no longer pinning the array endpoints. Ignored (with a warning)
        for any other method, since those already follow the N//2 convention
        (bilinear/bicubic/nearest via grid_sample's own centering, fourier via
        the same zero-pad/crop resize() used by fft/ifft).

    Returns:
    --------
    x : (torch.Tensor)
        The resized tensor with shape (..., *im_size)
    """

    if preserve_dc_position and method != "poly":
        warn(
            f"preserve_dc_position=True is only supported for method='poly' "
            f"(got method={method!r}); ignoring and proceeding with "
            f"preserve_dc_position=False."
        )
        preserve_dc_position = False

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
            resize_poly = lambda value: _spatial_resize_poly_cubic(
                value, im_size, preserve_dc_position=preserve_dc_position
            )
        else:
            resize_poly = lambda value: _spatial_resize_poly_lagrange(
                value, im_size, order=order, preserve_dc_position=preserve_dc_position
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
