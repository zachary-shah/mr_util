from math import sqrt
from typing import Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import windows

SPATIAL_RESIZE_METHODS = Literal["bilinear", "bicubic", "nearest"]
RESIZE_METHODS = Literal["bilinear", "bicubic", "nearest", "fourier"]
WINDOW_METHODS = Literal["boxcar", "hamming", "hann", "blackman", "kaiser"]
DEFAULT_WINDOW = "hann"


def resize(input: torch.Tensor, oshape: Tuple[int, ...]) -> torch.Tensor:
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

    output = torch.zeros(oshape, dtype=input.dtype, device=input.device)
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


def gen_grd(
    im_size: tuple, fovs: Optional[tuple] = None, balanced: Optional[bool] = False
) -> torch.Tensor:
    """
    Generates a grid of points given image size and FOVs

    Parameters:
    -----------
    im_size : tuple
        image dimensions
    fovs : tuple
        field of views, same size as im_size

    Returns:
    --------
    grd : torch.Tensor
        grid of points with shape (*im_size, len(im_size))
    """
    if fovs is None:
        fovs = (1,) * len(im_size)
    if balanced:
        lins = [
            fovs[i] * torch.linspace(-1 / 2, 1 / 2, im_size[i])
            for i in range(len(im_size))
        ]
    else:
        lins = [
            fovs[i]
            * torch.arange(-(im_size[i] // 2), im_size[i] // 2 + (im_size[i] % 2))
            / (im_size[i])
            for i in range(len(im_size))
        ]
    grds = torch.meshgrid(*lins, indexing="ij")
    grd = torch.cat([g[..., None] for g in grds], dim=-1)

    return grd.type(torch.float32)


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


def _spatial_resize(
    x: torch.Tensor,
    im_size: tuple,
    method: SPATIAL_RESIZE_METHODS = "bilinear",
) -> torch.Tensor:
    """
    Resize a spatial tensor to a new spatial size.

    Parameters:
    -----------
    x : (torch.Tensor)
        The input tensor with shape (B, *inp_im_size)
    im_size : (tuple)
        The size of the image to resize to
    method : (Optional[str])
        The method to use for resizing, options are:
            - 'bilinear': linear interpolation (default)
            - 'bicubic': cubic interpolation
            - 'nearest': nearest sample interpolation

    Returns:
    --------
    x_rs : (torch.Tensor)
        The resized tensor with shape (B, *im_size)
    """

    n_spatial = len(im_size)
    if n_spatial == 3 and method == "bicubic":
        warn(
            "Bicubic interpolation is not supported for 3D data, using bilinear instead."
        )
        method = "bilinear"

    grd = 2 * gen_grd(im_size, balanced=True).to(x.device).flip(-1)
    grd = grd[None].repeat_interleave(x.shape[0], dim=0)

    def gs(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.grid_sample(
            x.unsqueeze(1), grd, align_corners=True, mode=method
        ).squeeze(1)

    if torch.is_complex(x):
        x = gs(x.real) + 1j * gs(x.imag)
    else:
        x = gs(x)

    return x


def spatial_resize(
    x: torch.Tensor,
    im_size: tuple,
    method: RESIZE_METHODS = "bilinear",
    window: Optional[WINDOW_METHODS] = None,
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
    window : (Optional[str])
        The window to apply to the fourier interpolation

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
        x = _fourier_resize(x, im_size, window)
    elif method in SPATIAL_RESIZE_METHODS.__args__:
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


def spatial_filter(
    x: torch.Tensor,
    im_size: Tuple[int, ...],
    gaussian_filter_size: Optional[Union[float, Tuple[float, ...]]] = None,
    median_filter_size: Optional[Union[int, Tuple[int, ...]]] = None,
) -> torch.Tensor:
    """
    Apply spatial filters to image.

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
