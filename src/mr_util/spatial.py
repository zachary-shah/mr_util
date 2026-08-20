from typing import Literal, Optional
from warnings import warn

import torch

SPATIAL_RESIZE_METHODS = Literal["bilinear", "bicubic", "nearest"]
RESIZE_METHODS = Literal["bilinear", "bicubic", "nearest", "fourier", "poly"]
WINDOW_METHODS = Literal["boxcar", "hamming", "hann", "blackman", "kaiser"]
DEFAULT_WINDOW = "hann"

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - Triton is optional for CPU-only use.
    triton = None
    tl = None


if TRITON_AVAILABLE:
    @triton.jit
    def _cubic_spline_prefilter_kernel(
        data,
        n_signals: tl.constexpr,
        n_samples: tl.constexpr,
        block_size: tl.constexpr,
        n_boundary: tl.constexpr,
    ):
        signals = tl.program_id(0) * block_size + tl.arange(0, block_size)
        valid = signals < n_signals
        row = signals * n_samples
        z = -0.26794919243112270647

        c0 = tl.load(data + row, mask=valid)
        last = tl.load(data + row + n_samples - 1, mask=valid)
        z_n = tl.exp(n_samples * tl.log(-z))
        if n_samples % 2:
            z_n = -z_n
        initial = c0 + z_n * last
        z_i = z
        for i in tl.static_range(1, n_boundary):
            if i < n_samples:
                left = tl.load(data + row + i, mask=valid)
                right = tl.load(data + row + n_samples - 1 - i, mask=valid)
                initial += z_i * (left + z_n * right)
            z_i *= z
        initial = initial * z / (1.0 - z_n * z_n) + c0
        tl.store(data + row, initial, mask=valid)

        previous = initial
        for i in tl.range(1, n_samples):
            value = tl.load(data + row + i, mask=valid)
            value += z * previous
            tl.store(data + row + i, value, mask=valid)
            previous = value

        previous *= z / (z - 1.0)
        tl.store(data + row + n_samples - 1, previous, mask=valid)
        for i in tl.range(n_samples - 2, -1, -1):
            value = tl.load(data + row + i, mask=valid)
            value = z * (previous - value)
            tl.store(data + row + i, value, mask=valid)
            previous = value


def hilbert(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the Hilbert transform of a real-valued signal.

    Parameters:
    -----------
    x : torch.Tensor
        The real-valued signal to compute the Hilbert transform of.
    dim : int
        The dimension over which to compute the Hilbert transform.
        Default is -1.

    Returns:
    --------
    x : torch.Tensor
        The Hilbert transform of the signal.
    """
    N = x.shape[dim]
    Xf = torch.fft.fft(x, dim=dim)
    h = torch.zeros(N, dtype=x.dtype, device=x.device)
    if N % 2 == 0:
        h[0] = 1
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if x.ndim > 1:
        hind = [None,] * x.ndim
        hind[dim] = slice(None)
        h = h[tuple(hind)]
    x = torch.fft.ifft(Xf * h, dim=dim)
    return x


def rotation_matrix_3d(thetas: torch.Tensor, rot_order: str = "xyz") -> torch.Tensor:
    """
    From Daniel Abraham `mr_recon` 

    Computes product rotation matrices for a set of X Y Z rotation angles
    
    Parameters:
    -----------
    thetas : torch.Tensor
        angle of rotation in radians with shape (..., 3)
    rot_order : str
        order of rotations, string ordering of x, y, z
        Default is "xyz".

    Returns:
    --------
    R : torch.Tensor
        rotation matrix with shape (..., 3, 3)
    """
    tup = (None,) * (thetas.ndim - 1) + (slice(None),)
    Rxs = rotation_matrix(torch.tensor([1.0, 0, 0], 
                                       device=thetas.device, 
                                       dtype=thetas.dtype)[tup], 
                          thetas[..., 0])
    Rys = rotation_matrix(torch.tensor([0, 1.0, 0], 
                                       device=thetas.device, 
                                       dtype=thetas.dtype)[tup], 
                          thetas[..., 1])
    Rzs = rotation_matrix(torch.tensor([0, 0, 1.0], 
                                       device=thetas.device, 
                                       dtype=thetas.dtype)[tup], 
                          thetas[..., 2])

    rots = {
        "x": Rxs,
        "y": Rys,
        "z": Rzs,
    }

    return rots[rot_order[0]] @ rots[rot_order[1]] @ rots[rot_order[2]]


def rotation_matrix(axis: torch.Tensor, 
                    theta: torch.Tensor) -> torch.Tensor:
    """
    From Daniel Abraham `mr_recon` 

    Computes rotation matrices for a given axis and angle

    Parameters:
    -----------
    axis : torch.Tensor
        axis of rotation with shape (..., 3)
    theta : torch.Tensor
        angle of rotation in radians with shape (...)
    
    Returns:
    --------
    R : torch.Tensor
        rotation matrix with shape (..., 3, 3)
    """
    
    dev = axis.device
    axis = axis / torch.linalg.norm(axis, dim=-1)
    a = torch.cos(theta / 2.0)
    b = -axis[..., 0] * torch.sin(theta / 2.0)
    c = -axis[..., 1] * torch.sin(theta / 2.0)
    d = -axis[..., 2] * torch.sin(theta / 2.0)
    R = torch.zeros((*theta.shape, 3, 3), device=dev, dtype=torch.float32)
    R[..., 0, 0] = a * a + b * b - c * c - d * d
    R[..., 0, 1] = 2 * (b * c - a * d)
    R[..., 0, 2] = 2 * (b * d + a * c)
    R[..., 1, 0] = 2 * (b * c + a * d)
    R[..., 1, 1] = a * a + c * c - b * b - d * d
    R[..., 1, 2] = 2 * (c * d - a * b)
    R[..., 2, 0] = 2 * (b * d - a * c)
    R[..., 2, 1] = 2 * (c * d + a * b)
    R[..., 2, 2] = a * a + d * d - b * b - c * c
    return R


def gen_grd(
    im_size: tuple, fovs: Optional[tuple] = None, balanced: Optional[bool] = False,
    device: Optional[torch.device] = None
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
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
        
    if fovs is None:
        fovs = (1,) * len(im_size)
    if balanced:
        lins = [
            fovs[i] * torch.linspace(-1 / 2, 1 / 2, im_size[i], **kwargs)
            for i in range(len(im_size))
        ]
    else:
        lins = [
            fovs[i]
            * torch.arange(-(im_size[i] // 2), im_size[i] // 2 + (im_size[i] % 2), **kwargs)
            / (im_size[i])
            for i in range(len(im_size))
        ]
    grds = torch.meshgrid(*lins, indexing="ij")
    grd = torch.cat([g[..., None] for g in grds], dim=-1)

    return grd.type(torch.float32)


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

    grd = 2 * gen_grd(im_size, balanced=True, device=x.device).flip(-1)
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


def _poly_resize_coords(
    in_size: int,
    out_size: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
    preserve_dc_position: bool = False,
) -> torch.Tensor:
    """
    Output-index -> input-coordinate map shared by the poly resize interpolants.

    preserve_dc_position=False (default, current behavior): align-corners, pins
    array endpoints (index 0 <-> 0, index N-1 <-> N-1). This only coincides with
    the N//2 DC/FFT-center convention used elsewhere in this codebase (gen_grd,
    fft, ifft, resize) when BOTH in_size and out_size are odd -- for any resize
    where one or both sizes are even, this introduces a sub-pixel offset between
    the resized array's DC pixel and the N//2 convention.

    preserve_dc_position=True: maps output index out_size//2 to input index
    in_size//2 exactly (scaled by in_size/out_size around that point), matching
    the DC convention used everywhere else, at the cost of no longer pinning the
    array endpoints (coordinates can fall slightly outside [0, in_size-1] near
    the edges; callers must handle/clamp that).
    """
    if out_size == 1:
        return torch.zeros(1, device=device, dtype=dtype)
    j = torch.arange(out_size, device=device, dtype=dtype)
    if preserve_dc_position:
        return (j - out_size // 2) * (in_size / out_size) + (in_size // 2)
    return j * ((in_size - 1) / (out_size - 1))


def _spatial_resize_poly_lagrange(
    x: torch.Tensor,
    im_size: tuple,
    order: Optional[int] = 3,
    preserve_dc_position: bool = False,
) -> torch.Tensor:
    """Resize a spatial tensor to a new spatial size using Lagrange-weighted polynomial interpolation."""
    if order is None:
        order = 3
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a non-negative integer")
    if len(im_size) != x.ndim - 1:
        raise ValueError("im_size must contain one entry per spatial dimension")
    if any(size <= 0 for size in im_size):
        raise ValueError("all output dimensions must be positive")

    # Interpolate one axis at a time. Polynomial interpolation is separable, and
    # this avoids constructing the product stencil of all spatial dimensions.
    for dim, out_size in enumerate(im_size, start=1):
        in_size = x.shape[dim]
        if in_size == out_size:
            continue

        degree = min(order, in_size - 1)
        support = degree + 1
        coord_dtype = (
            torch.float64 if x.dtype in (torch.float64, torch.complex128)
            else torch.float32
        )
        coords = _poly_resize_coords(
            in_size, out_size, coord_dtype, x.device, preserve_dc_position
        )

        if degree == 0:
            indices = torch.floor(coords + 0.5).to(torch.long)
            x = x.index_select(dim, indices)
            continue

        # Use a centered, contiguous stencil and shift it at the boundaries.
        starts = torch.floor(coords).to(torch.long) - degree // 2
        starts.clamp_(0, in_size - support)
        offsets = torch.arange(support, device=x.device)
        nodes = starts[:, None] + offsets[None, :]

        # Lagrange basis weights for each output coordinate.
        delta = coords[:, None] - nodes.to(coord_dtype)
        weights = torch.ones_like(delta)
        for j in range(support):
            for k in range(support):
                if j != k:
                    weights[:, j].mul_(delta[:, k] / (j - k))

        # Gather all stencils in one operation, then reduce the support axis.
        values = x.index_select(dim, nodes.reshape(-1))
        shape = list(values.shape)
        shape[dim : dim + 1] = [out_size, support]
        values = values.reshape(shape)
        weight_shape = [1] * values.ndim
        weight_shape[dim : dim + 2] = [out_size, support]
        x = (values * weights.reshape(weight_shape)).sum(dim=dim + 1)

    return x


def _cubic_spline_filter_axis(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Convert samples along one axis to cubic B-spline coefficients."""
    if torch.is_complex(x):
        return torch.complex(
            _cubic_spline_filter_axis(x.real, dim),
            _cubic_spline_filter_axis(x.imag, dim),
        )

    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    lines = x.movedim(dim, -1).contiguous().to(dtype)
    shape = lines.shape
    lines = lines.reshape(-1, shape[-1])
    z = -0.26794919243112270647

    if lines.is_cuda and TRITON_AVAILABLE and not x.requires_grad:
        lines = lines * 6.0
        n_signals, n_samples = lines.shape
        block_size = 128
        _cubic_spline_prefilter_kernel[(triton.cdiv(n_signals, block_size),)](
            lines,
            n_signals=n_signals,
            n_samples=n_samples,
            block_size=block_size,
            n_boundary=32 if dtype == torch.float64 else 18,
        )
    else:
        # Differentiable fallback matching SciPy's reflect-boundary IIR.
        lines = lines * 6.0
        n_samples = lines.shape[-1]
        z_n = z ** n_samples
        boundary = min(n_samples, 32 if dtype == torch.float64 else 18)
        initial = lines[:, 0] + z_n * lines[:, -1]
        if boundary > 1:
            i = torch.arange(1, boundary, device=x.device, dtype=dtype)
            powers = torch.pow(torch.as_tensor(z, device=x.device, dtype=dtype), i)
            reflected = lines[:, -boundary:-1].flip(-1)
            initial = initial + (
                powers[None]
                * (lines[:, 1:boundary] + z_n * reflected)
            ).sum(-1)
        initial = initial * z / (1.0 - z_n * z_n) + lines[:, 0]

        causal = [initial]
        for i in range(1, n_samples):
            causal.append(lines[:, i] + z * causal[-1])
        causal = torch.stack(causal, dim=-1)

        anticausal = [causal[:, -1] * z / (z - 1.0)]
        for i in range(n_samples - 2, -1, -1):
            anticausal.append(z * (anticausal[-1] - causal[:, i]))
        lines = torch.stack(anticausal[::-1], dim=-1)

    return lines.reshape(shape).movedim(-1, dim)


def _spatial_resize_poly_cubic(
    x: torch.Tensor,
    im_size: tuple,
    preserve_dc_position: bool = False,
) -> torch.Tensor:
    """Resize with the prefiltered cubic B-spline used by map_coordinates."""
    if len(im_size) != x.ndim - 1:
        raise ValueError("im_size must contain one entry per spatial dimension")
    if any(size <= 0 for size in im_size):
        raise ValueError("all output dimensions must be positive")

    pad = 12
    inp_size = x.shape[1:]

    # map_coordinates pads mode="nearest" inputs before spline filtering.
    for dim in range(1, x.ndim):
        first = x.select(dim, 0).unsqueeze(dim)
        last = x.select(dim, x.shape[dim] - 1).unsqueeze(dim)
        expand_shape = list(x.shape)
        expand_shape[dim] = pad
        x = torch.cat(
            (first.expand(expand_shape), x, last.expand(expand_shape)), dim=dim
        )

    for dim in range(1, x.ndim):
        x = _cubic_spline_filter_axis(x, dim)

    coord_dtype = (
        torch.float64 if x.dtype in (torch.float64, torch.complex128)
        else torch.float32
    )
    for dim, (in_size, out_size) in enumerate(zip(inp_size, im_size), start=1):
        coords = _poly_resize_coords(
            in_size, out_size, coord_dtype, x.device, preserve_dc_position
        )
        coords.add_(pad)

        base = torch.floor(coords).to(torch.long)
        t = coords - base.to(coord_dtype)
        # preserve_dc_position maps don't pin array endpoints, so coordinates can
        # fall slightly outside the padded input near the edges; clamp node
        # indices into the valid (padded) range rather than index out of bounds.
        nodes = (base[:, None] + torch.arange(-1, 3, device=x.device)[None]).clamp(
            0, x.shape[dim] - 1
        )
        weights = torch.stack(
            (
                (1.0 - t) ** 3,
                3.0 * t**3 - 6.0 * t**2 + 4.0,
                -3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0,
                t**3,
            ),
            dim=-1,
        ) / 6.0

        values = x.index_select(dim, nodes.reshape(-1))
        shape = list(values.shape)
        shape[dim : dim + 1] = [out_size, 4]
        values = values.reshape(shape)
        weight_shape = [1] * values.ndim
        weight_shape[dim : dim + 2] = [out_size, 4]
        x = (values * weights.reshape(weight_shape)).sum(dim=dim + 1)

    return x

