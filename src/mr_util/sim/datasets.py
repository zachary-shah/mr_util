import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download

from ..sim import paths
from ..utils import (
    RESIZE_METHODS,
    WINDOW_METHODS,
    gen_grd,
    resize,
    spatial_filter,
    spatial_resize,
)


class QuantitativeDataset:
    def __init__(
        self,
        fov: torch.Tensor,
        PD: torch.Tensor,
        T2s: Optional[torch.Tensor] = None,
        T2: Optional[torch.Tensor] = None,
        T1: Optional[torch.Tensor] = None,
        B1: Optional[torch.Tensor] = None,
        B0: Optional[torch.Tensor] = None,
        mps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        iso: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        """
        Set up a quantitative imaging dataset with the provided tensors.

        Parameters
        ----------
        fov : torch.Tensor
            Field of view (FOV) in meters, shape (ndim,).
        PD : torch.Tensor
            Proton Density map tensor, shape (*im_size).
        T2s : Optional[torch.Tensor]
            T2* map in seconds, shape (*im_size). If None, will not be used
        T2 : Optional[torch.Tensor]
            T2 map in seconds, shape (*im_size). If None, will not be used
        T1 : Optional[torch.Tensor]
            T1 map in seconds, shape (*im_size). If None, will not be used
        B1 : Optional[torch.Tensor]
            B1 map for transmit inhomogeneities. TODO: add support for B1 in acquired model.
        B0 : Optional[torch.Tensor]
            B0 map in Hz, shape (*im_size). If None, will not be used
        mps : Optional[torch.Tensor]
            Sensitivity maps tensor, shape (Nc, *im_size), where Nc is the number of coils
        mask : Optional[torch.Tensor]
            Binary mask tensor, shape (*im_size). If None, will not be used
        iso : Optional[torch.Tensor]
            Center position of the FOV in the image, length (3,) in meters.
            Can be length (2,) for 2D datasets, in which case the third dimension is assumed to be 0.
        eps : float
            Small value to avoid division by zero in T2* and T2 calculations. Default is 1e-8.

        Additional Attributes Created
        -----------------------------
        coords : torch.Tensor
            Coordinates tensor of shape (*im_size, ndim) containing the coordinates of each voxel in the
            image, adjusted by the iso position.

        Relevant Methods
        ----------------
        `evaluate(tt, ...)` -> torch.Tensor
            Evaluate the signal at times `tt` for the specified signal type (GRE or SE).
        `resize_fov(new_fov, corner=None)` -> None
            Adjust the field of view (FOV) of the dataset, with optional crop or padding
        `resize_matrix(new_im_size, ...)` -> None
            Resize the image matrix to the new size, using specified resizing methods.
        `transpose(dims)` -> None
            Transpose the spatial dimensions of the dataset.
        `flip(dims)` -> None
            Apply flips on desired dimensions to all items in the dataset.
        `set_slice(spatial_slice)` -> None
            Create a slice on the dataset along the specified dimensions.
            Useful for sub-selecting slices for 3D datasets.
        `get_maps()` -> Dict[str, torch.Tensor]
            Return a dictionary of all available maps in the dataset.
        `get_coords()` -> torch.Tensor
            Return the coordinates tensor of the dataset.
        `get_fov()` -> torch.Tensor
            Return the field of view tensor of the dataset.
        """

        self.fov = fov
        self.ndim = len(fov)
        assert self.ndim in [2, 3], "Quantitative Image dimensions must be 2D or 3D."

        if iso is not None:
            if not isinstance(iso, torch.Tensor):
                iso = torch.tensor(iso, device=fov.device, dtype=torch.float32)
            if self.ndim == 2 and len(iso) == 2:
                iso = torch.tensor(
                    [iso[0].item(), iso[1].item(), 0.0],
                    device=fov.device,
                    dtype=torch.float32,
                )
            assert len(iso) == 3, "iso position must be length 3."
            self.iso = iso.to(self.device).to(torch.float32)
        else:
            self.iso = torch.zeros((3,), device=fov.device, dtype=torch.float32)

        self.im_size = PD.shape
        self.device = PD.device
        assert (
            PD.ndim == self.ndim
        ), "Spatial maps must have the same number of dimensions as fov."

        self.PD = PD
        self.T2s = self.__validate_tensor(T2s, "T2s")
        self.T2 = self.__validate_tensor(T2, "T2")
        self.T1 = self.__validate_tensor(T1, "T1")
        self.B1 = self.__validate_tensor(B1, "B1")
        self.B0 = self.__validate_tensor(B0, "B0")
        self.mps = self.__validate_tensor(mps, "mps")
        self.mask = self.__validate_tensor(mask, "mask")
        self.eps = eps
        if self.mps is not None:
            self.Nc = self.mps.shape[0]

        # for retrieving spatial subset of data
        self.spatial_slice = None

        self.coords = self._update_coordinates()

    def __validate_tensor(self, x, name):
        if x is not None:
            assert (
                x.shape[-self.ndim :] == self.im_size
            ), f"Shape mismatch for {name}: expected {self.im_size}, got {x.shape[-self.ndim:]}"
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device)
        return x

    def _update_coordinates(self, iso: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Form a (*im_size, 3) tensor containing 3D coordinates of each voxel in image.
        """

        fov_ = self.fov.cpu().numpy().tolist()
        im_size_ = self.im_size

        if self.ndim == 2:
            fov_ = [fov_[0], fov_[1], 1.0]
            im_size_ = (im_size_[0], im_size_[1], 1)

        coords = gen_grd(im_size_, fovs=fov_)

        if self.ndim == 2:
            coords = coords[:, :, 0]

        coords = coords.to(torch.float32).to(self.device)

        # adjust to center position
        if iso is None:
            iso = self.iso.clone()
        assert len(iso) == 3, "provided iso position must be length 3 and in meters."
        self.iso = iso.clone()

        for _ in range(self.ndim):
            iso = iso[
                None,
            ]
        coords = coords + iso

        # update attributes
        self.coords = coords

        if self.spatial_slice is not None:
            coords = coords[(*self.spatial_slice, slice(None))]

        return coords

    def _evaluate_baseline_signal(self, TR: Optional[float] = None) -> torch.Tensor:
        if self.spatial_slice is not None:
            slc = (None,) + self.spatial_slice
        else:
            slc = (None,)

        x = self.PD[slc]
        if self.T1 is not None and TR is not None:
            x = x * (1 - torch.exp(-TR / self.T1[slc]))
        return x

    def _apply_b0(self, x: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        if self.spatial_slice is not None:
            slc = (None,) + self.spatial_slice
        else:
            slc = (None,)

        if self.B0 is not None:
            x = x.to(torch.complex64)
            x = x * torch.exp(-1j * 2 * torch.pi * self.B0[slc] * tt)
        return x

    def _evaluate_gre(
        self, tt: torch.Tensor, TR: Optional[float] = None
    ) -> torch.Tensor:
        """
        Evaluate the GRE signal at times provided in `tt`.

        Parameters
        ----------
        tt : torch.Tensor
            Time points at which to evaluate the GRE signal.

        Returns
        -------
            complex timepoint images as shape (Nt, *im_size)
        """
        assert self.T2s is not None, "T2s must be provided to evaluate GRE signal."

        if self.spatial_slice is not None:
            slc = (None,) + self.spatial_slice
        else:
            slc = (None,)

        # add broadcase dims
        for _ in range(self.ndim):
            tt = tt[..., None]

        x = self._evaluate_baseline_signal(TR)
        x = x * torch.exp(-tt / self.T2s[slc]).to(x.dtype)
        x = self._apply_b0(x, tt)
        return x

    def _evaluate_se(
        self, tt: torch.Tensor, TE: float, TR: Optional[float] = None
    ) -> torch.Tensor:
        """
        Evaluate a Spin Echo signal at times provided in `tt`.

        Parameters
        ----------
        tt : torch.Tensor
            Times (seconds) since excitation at which to evaluate the SE signal.
        TE : float
            Time (seconds) at which the signal is refocused, relative to the excitation time.

        Returns
        -------
            complex timepoint images as shape (Nt, *im_size)
        """
        assert self.T2 is not None, "T2 must be provided to evaluate SE signal."
        assert self.T2s is not None, "T2s must be provided to evaluate SE signal."

        if self.spatial_slice is not None:
            slc = (None,) + self.spatial_slice
        else:
            slc = (None,)

        T2pinv = 1 / (self.T2s[slc] + self.eps) - 1 / (self.T2[slc] + self.eps)

        x = self._evaluate_baseline_signal(TR)

        x = x.repeat_interleave(tt.shape[0], dim=0)

        if self.B0 is not None:
            x = x.to(torch.complex64)

        # less than refoc --> becomes gradient echo
        refoc_time = TE / 2
        gre_inds = tt < refoc_time
        se_inds = ~gre_inds

        # add broadcase dims
        for _ in range(self.ndim):
            tt = tt[..., None]

        if gre_inds.any():
            x[gre_inds] = x[gre_inds] * torch.exp(-tt[gre_inds] / self.T2s[slc]).to(
                x.dtype
            )
            x[gre_inds] = self._apply_b0(x[gre_inds], tt[gre_inds])
        if se_inds.any():
            tt_se = tt[se_inds]
            x[se_inds] = x[se_inds] * torch.exp(
                -((tt_se - TE).abs() * T2pinv) - (tt_se / self.T2[slc])
            ).to(x.dtype)
            x[se_inds] = self._apply_b0(x[se_inds], tt_se - TE)

        return x

    def evaluate(
        self,
        tt: Union[float, torch.Tensor],
        signal_type: Literal["gre", "se"] = "gre",
        TE: Optional[float] = None,
        TR: Optional[float] = None,
        apply_maps: bool = False,
        apply_mask: bool = False,
    ) -> torch.Tensor:
        """
        Evaluate the signal at times `tt` for the specified signal type.

        Parameters
        ----------
        tt : Union[float, torch.Tensor]
            Time points at which to evaluate the signal. If a float, will return data with shape im_size,
            otherwise will return data with shape (Nt, *im_size).
        signal_type : Literal["gre", "se"]
            Type of signal to evaluate, either "gre" for Gradient Echo or "se" for Spin Echo.
        TE : Optional[float]
            Time at which the Spin Echo signal is refocused, relative to the excitation time.
            Required for signal_type "se".
        TR : Optional[float]
            Repetition time for the signal, required for both signal types.
            Used for T1 application.

        Returns
        -------
        torch.Tensor
            Evaluated signal at the specified times, with shape (Nt, *im_size) if `tt` is a tensor,
            or with shape (*im_size) if `tt` is a float.
        """

        assert signal_type in ["gre", "se"], "signal_type must be either 'gre' or 'se'."
        if signal_type == "se":
            assert (
                TE is not None
            ), "TE must be provided for Spin Echo signal evaluation."

        float_input = isinstance(tt, float)
        if float_input:
            tt = torch.tensor([tt], device=self.device, dtype=torch.float32)

        if signal_type == "gre":
            x = self._evaluate_gre(tt, TR)
        elif signal_type == "se":
            x = self._evaluate_se(tt, TE, TR)

        if apply_mask:
            assert self.mask is not None, "mask must be provided to apply mask."
            if self.spatial_slice is not None:
                slc = (None,) + self.spatial_slice
            else:
                slc = (None,)
            x = x * self.mask[slc]

        if apply_maps:
            assert (
                self.mps is not None
            ), "sensitivity maps must be provided to apply maps."
            if self.spatial_slice is not None:
                slc = (
                    (None,)
                    + (
                        slice(
                            None,
                        ),
                    )
                    + self.spatial_slice
                )
            else:
                slc = (None,)

            x = x[:, None] * self.mps[slc]

        if float_input:
            x = x.squeeze(0)

        return x

    def resize_matrix(
        self,
        new_im_size: Tuple[int, ...],
        method: RESIZE_METHODS = "bilinear",
        window: Optional[WINDOW_METHODS] = None,
    ) -> None:
        """
        Resize the image matrix to the new size. See `mr_util.utils.spatial_resize` for details
        on resizing methods arguments.

        Parameters
        ----------
        new_im_size : Tuple[int, ...]
            The new image size to set for the dataset.
        method : RESIZE_METHODS
            Method to use for resizing the dataset, defaults to "bilinear".
        window : Optional[WINDOW_METHODS]
            Method to use for windowing the dataset during resizing, defaults to None.
        """

        if self.spatial_slice is not None:
            raise ValueError(
                "Cannot resize matrix with a spatial slice set. Clear the slice first."
            )

        self.im_size = new_im_size
        self.coords = self._update_coordinates()

        def resize_wrapper(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is not None:
                return spatial_resize(
                    x,
                    new_im_size,
                    method=method,
                    window=window,
                )
            return None

        self.PD = resize_wrapper(self.PD)
        self.T2s = resize_wrapper(self.T2s)
        self.T2 = resize_wrapper(self.T2)
        self.T1 = resize_wrapper(self.T1)
        self.B1 = resize_wrapper(self.B1)
        self.B0 = resize_wrapper(self.B0)
        self.mps = resize_wrapper(self.mps)
        self.mask = resize_wrapper(self.mask)

    def resize_fov(
        self,
        new_fov: Union[Tuple[float, ...], torch.Tensor],
        corner: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
    ) -> None:
        """
        Adjust the field of view (FOV) of the dataset.

        If new_fov is greater than current fov, the dataset will be padded with zeros.
        Otherwise, the dataset will be cropped to the new fov, with optional anchoring corner.

        Parameters
        ----------
        new_fov : Union[Tuple[float, ...], torch.Tensor]
            The new field of view to set for the dataset.
        corner : Optional[Union[Tuple[float, ...], torch.Tensor]]
            The corner position, in meters, from which to crop the dataset, if new_fov is smaller
            than current fov. If None, the crop defaults to the center of the current fov.
            If provided and corner=-1 for any dimension, that dimension will remain center cropped.
        """

        assert self.spatial_slice is None, "Cannot resize FOV with a spatial slice set."

        if not isinstance(new_fov, torch.Tensor):
            new_fov = torch.tensor(new_fov, device=self.device, dtype=torch.float32)

        old_im_size = torch.tensor(
            self.im_size, device=self.device, dtype=torch.float32
        )
        new_im_size = tuple(
            (old_im_size * (new_fov / self.fov)).cpu().numpy().round().astype(int)
        )
        new_iso = self.iso.clone()

        if (new_im_size == self.im_size) and (corner is None):
            return

        if corner is not None:
            if isinstance(corner, torch.Tensor):
                corner = corner.cpu().numpy().tolist()
            elif isinstance(corner, tuple):
                corner = list(corner)
            assert (
                len(corner) == self.ndim
            ), "corner must have the same number of dimensions as fov."

            for d in range(self.ndim):
                if corner[d] > 0:
                    new_iso[d] = corner[d] + (new_fov[d] / 2)
                    corner[d] = int(
                        round(corner[d] / self.fov[d].item() * old_im_size[d].item())
                    )
                    print(f"corner[{d}] = {corner[d]}")

        def resize_wrapper(
            x: Optional[torch.Tensor], dim_ofs=0
        ) -> Optional[torch.Tensor]:
            if x is not None:
                sz = list(new_im_size)
                if dim_ofs > 0:
                    sz = list(x.shape[:dim_ofs]) + sz
                if corner is not None:
                    for d in range(self.ndim):
                        if corner[d] >= 0:
                            # zero pad end of dim d + ofs
                            if (old_im_size[d] - corner[d]) < sz[d + dim_ofs]:
                                pad_shape = list(x.shape)
                                pad_shape[d + dim_ofs] = sz[d + dim_ofs] - (
                                    old_im_size[d] - corner[d]
                                )
                                zz = torch.zeros(
                                    tuple(pad_shape), device=x.device, dtype=x.dtype
                                )
                                x = torch.cat((x, zz), dim=d + dim_ofs)
                            # crop now, so no more resize on this dimension
                            x = x.narrow(d + dim_ofs, corner[d], sz[d + dim_ofs])
                            sz[d + dim_ofs] = x.shape[d + dim_ofs]
                return resize(x, tuple(sz))
            return None

        self.PD = resize_wrapper(self.PD)
        self.T2s = resize_wrapper(self.T2s)
        self.T2 = resize_wrapper(self.T2)
        self.T1 = resize_wrapper(self.T1)
        self.B1 = resize_wrapper(self.B1)
        self.B0 = resize_wrapper(self.B0)
        self.mps = resize_wrapper(self.mps, dim_ofs=1)
        self.mask = resize_wrapper(self.mask)

        # update fov and coords
        self.fov = new_fov
        self.im_size = new_im_size
        self.coords = self._update_coordinates(iso=new_iso)

    def transpose(self, dims: Tuple[int, ...]) -> None:
        """
        Tranpose the spatial dimensions of dataset.

        Parameters
        ----------
        dims : Tuple[int, ...]
            Desired permutation of spatial dimensions. Must have same number of dimensions as fov.

        """

        assert (
            len(dims) == self.ndim
        ), "dims must have the same number of dimensions as fov."

        def transpose_wrapper(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is not None:
                if x.ndim == self.ndim:
                    return x.permute(dims)
                elif x.ndim > self.ndim:
                    ndim_ofs = x.ndim - self.ndim
                    leading_dims = list(range(ndim_ofs))
                    return x.permute((*leading_dims, *[d + ndim_ofs for d in dims]))
                else:
                    raise ValueError(f"Cannot transpose tensor with shape {x.shape}.")
            return None

        self.PD = transpose_wrapper(self.PD)
        self.T2s = transpose_wrapper(self.T2s)
        self.T2 = transpose_wrapper(self.T2)
        self.T1 = transpose_wrapper(self.T1)
        self.B1 = transpose_wrapper(self.B1)
        self.B0 = transpose_wrapper(self.B0)
        self.mps = transpose_wrapper(self.mps)
        self.mask = transpose_wrapper(self.mask)

        # update spatial tracking
        self.im_size = tuple(self.im_size[d] for d in dims)
        self.fov = torch.tensor(
            tuple(self.fov[d].item() for d in dims),
            device=self.device,
            dtype=torch.float32,
        )
        if self.ndim == 2:
            if dims == (1, 0):
                self.iso = torch.tensor(
                    [self.iso[1].item(), self.iso[0].item(), self.iso[2].item()],
                    device=self.device,
                    dtype=torch.float32,
                )
        else:
            self.iso = torch.tensor(
                [self.iso[d].item() for d in dims],
                device=self.device,
                dtype=torch.float32,
            )
        self.coords = self._update_coordinates()

    def flip(self, dims: Tuple[int, ...]) -> None:
        """
        Apply flips on desired dimensions to all items in dataset

        Parameters
        ----------
        dims : Tuple[int, ...]
            Dimensions to flip. Can be negative to indicate from the end.
        """

        def flip_wrapper(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is not None:
                nofs = x.ndim - self.ndim
                for d in dims:
                    if d < 0:
                        x = torch.flip(x, dims=(d,))
                    else:
                        x = torch.flip(x, dims=(d + nofs,))
                return x.contiguous()
            return x

        self.PD = flip_wrapper(self.PD)
        self.T2s = flip_wrapper(self.T2s)
        self.T2 = flip_wrapper(self.T2)
        self.T1 = flip_wrapper(self.T1)
        self.B1 = flip_wrapper(self.B1)
        self.B0 = flip_wrapper(self.B0)
        self.mps = flip_wrapper(self.mps)
        self.mask = flip_wrapper(self.mask)
        self.coords = self._update_coordinates()

    def set_slice(self, spatial_slice: Optional[Tuple[slice, ...]]) -> None:
        """
        Create a slice on the dataset along the specified dimensions.

        Parameters
        ----------
        slices : Tuple[slice, ...]
            Slices to apply to each dimension of the dataset.
        """

        if spatial_slice is not None:
            # assert tuple with length same as ndim
            assert (
                len(spatial_slice) == self.ndim
            ), "spatial_slice must have the same number of dimensions as fov."
            assert isinstance(spatial_slice, tuple), "spatial_slice must be a tuple."
            for s in spatial_slice:
                assert isinstance(
                    s, (slice, list)
                ), "spatial_slice must contain slices."
        self.spatial_slice = spatial_slice
        self._update_coordinates()

    def get_maps(
        self, return_maps: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary of all available maps in the dataset.
        """
        maps = {}
        if self.PD is not None and (return_maps is None or "PD" in return_maps):
            if self.spatial_slice is not None:
                maps["PD"] = self.PD[self.spatial_slice].clone()
            else:
                maps["PD"] = self.PD.clone()
        if self.T2s is not None and (return_maps is None or "T2s" in return_maps):
            if self.spatial_slice is not None:
                maps["T2s"] = self.T2s[self.spatial_slice].clone()
            else:
                maps["T2s"] = self.T2s.clone()
        if self.T2 is not None and (return_maps is None or "T2" in return_maps):
            if self.spatial_slice is not None:
                maps["T2"] = self.T2[self.spatial_slice].clone()
            else:
                maps["T2"] = self.T2.clone()
        if self.T1 is not None and (return_maps is None or "T1" in return_maps):
            if self.spatial_slice is not None:
                maps["T1"] = self.T1[self.spatial_slice].clone()
            else:
                maps["T1"] = self.T1.clone()
        if self.B1 is not None and (return_maps is None or "B1" in return_maps):
            if self.spatial_slice is not None:
                maps["B1"] = self.B1[self.spatial_slice].clone()
            else:
                maps["B1"] = self.B1.clone()
        if self.B0 is not None and (return_maps is None or "B0" in return_maps):
            if self.spatial_slice is not None:
                maps["B0"] = self.B0[self.spatial_slice].clone()
            else:
                maps["B0"] = self.B0.clone()
        if self.mps is not None and (return_maps is None or "mps" in return_maps):
            if self.spatial_slice is not None:
                maps["mps"] = self.mps[(slice(None),) + self.spatial_slice].clone()
            else:
                maps["mps"] = self.mps.clone()
        if self.mask is not None and (return_maps is None or "mask" in return_maps):
            if self.spatial_slice is not None:
                maps["mask"] = self.mask[self.spatial_slice].clone()
            else:
                maps["mask"] = self.mask.clone()

        return maps

    def get_coords(self) -> torch.Tensor:
        """
        Return the coordinates tensor of the dataset.
        """
        if self.spatial_slice is not None:
            return self.coords[self.spatial_slice + (slice(None),)].clone()
        return self.coords.clone()

    def get_fov(self) -> torch.Tensor:
        """
        Return the field of view tensor of the dataset.
        """
        fov_out = self.fov.clone()
        if self.spatial_slice is not None:
            for d in range(self.ndim):
                if isinstance(self.spatial_slice[d], slice):
                    start = self.spatial_slice[d].start or 0
                    stop = self.spatial_slice[d].stop or self.im_size[d]
                    fov_out[d] = self.fov[d] * (stop - start) / self.im_size[d]
                elif isinstance(self.spatial_slice[d], list):
                    inds = np.array(self.spatial_slice[d])
                    width = inds.max() - inds.min() + 1
                    fov_out[d] = self.fov[d] * width / self.im_size[d]
        return fov_out

    def to(self, device: torch.DeviceObjType):
        """
        Move the dataset to a specified device.

        Parameters
        ----------
        device : torch.DeviceObjType
            The device to move the dataset to.

        Returns
        -------
        QuantitativeDataset
            A new instance of QuantitativeDataset on the specified device.
        """

        def to_wrapper(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is not None:
                return x.to(device)
            return None

        self.PD = to_wrapper(self.PD)
        self.T2s = to_wrapper(self.T2s)
        self.T2 = to_wrapper(self.T2)
        self.T1 = to_wrapper(self.T1)
        self.B1 = to_wrapper(self.B1)
        self.B0 = to_wrapper(self.B0)
        self.mps = to_wrapper(self.mps)
        self.mask = to_wrapper(self.mask)
        self.fov = self.fov.to(device)
        self.iso = self.iso.to(device)
        self.coords = self.coords.to(device)
        self.device = device

        return self


class SimDataset(Enum):
    BRAIN_3D = "Brain_3D"
    HEAD_3D = "Head_3D"
    AXIAL_2D = "Axial_2D"


@dataclass
class MapFilterConfig:
    clip_min: Optional[float] = None  # seconds
    clip_max: Optional[float] = None  # seconds
    gaussian_std: Optional[Union[float, Tuple[float, ...]]] = None
    median_filter_size: Optional[Union[int, Tuple[int, ...]]] = None


def apply_filter(x: torch.Tensor, cfg: MapFilterConfig) -> torch.Tensor:
    if (cfg.clip_min is not None) or (cfg.clip_max is not None):
        if cfg.clip_min is None:
            cfg.clip_min = x.min()
        if cfg.clip_max is None:
            cfg.clip_max = x.max()
        x = torch.clamp(x, min=cfg.clip_min, max=cfg.clip_max)

    if (cfg.gaussian_std is not None) or (cfg.median_filter_size is not None):
        x = spatial_filter(
            x,
            x.shape,
            gaussian_filter_size=cfg.gaussian_std,
            median_filter_size=cfg.median_filter_size,
        )
    return x


@dataclass
class DatasetFilterConfig:
    pd_cfg: MapFilterConfig = MapFilterConfig()
    t1_cfg: MapFilterConfig = MapFilterConfig(clip_min=0.0, clip_max=3.0)
    t2_cfg: MapFilterConfig = MapFilterConfig(clip_min=0.0, clip_max=0.5)
    t2s_cfg: MapFilterConfig = MapFilterConfig(clip_min=0.0, clip_max=0.25)
    b0_cfg: MapFilterConfig = MapFilterConfig()


def get_hf_dataset(
    dataset: SimDataset,
    device: Optional[torch.DeviceObjType] = None,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    assert isinstance(
        dataset, SimDataset
    ), "dataset must be an instance of SimDataset Enum."
    dataset_str = dataset.value
    os.makedirs(paths.SIM_DATA_DIR, exist_ok=True)
    file_name = paths.SIM_DATASETS[dataset_str]
    file_path = paths.SIM_DATA_DIR / file_name

    if device is None:
        device = torch.device("cpu")

    if os.path.exists(file_path):
        if verbose:
            print(f"Loading {dataset_str} from local path: {file_path}")
        data = torch.load(file_path, map_location=device, weights_only=True)
    else:
        # retrieve from Hf hub
        if verbose:
            print(
                f"Loading {dataset_str} from Hugging Face Hub: {paths.HF_REPO}/{file_name}"
            )
        snapshot_download(
            repo_id=paths.HF_REPO,
            local_dir=paths.SIM_DATA_DIR,
            allow_patterns=[file_name],
            repo_type="dataset",
        )
        data = torch.load(file_path, map_location=device, weights_only=True)

    if verbose:
        print(f"Loaded data with keys: {list(data.keys())}")

    return data


def load_dataset(
    dataset: SimDataset,
    device: Optional[torch.DeviceObjType] = None,
    filters: Optional[DatasetFilterConfig] = None,
    im_size: Optional[Tuple[int, ...]] = None,
    resize_method: RESIZE_METHODS = "bilinear",
    window_method: WINDOW_METHODS = None,
    verbose: bool = False,
) -> QuantitativeDataset:
    """
    Load simulation dataset. Available datasets are:
    - Brain_3D : XYZ dataset just of masked brain
    - Head_3D : XYZ dataset over full head
    - Axial_2D : XY single slice dataset

    Parameters
    ----------
    dataset : SimDataset
        The dataset to load, must be an instance of SimDataset Enum.
    device : Optional[torch.DeviceObjType]
        Device to load the dataset onto, defaults to None (CPU).
    filters : Optional[DatasetFilterConfig]
        Configuration for filtering quantitative maps. See `MapFilterConfig` for details.
    im_size : Optional[Tuple[int, ...]]
        Desired output image size. If provided, the dataset will be resized to this size.
    resize_method : RESIZE_METHODS
        Method to use for resizing the dataset, defaults to "bilinear".
    window_method : Optional[WINDOW_METHODS]
        For fourier resizing

    Returns
    -------
    QuantitativeDataset
        An instance of QuantitativeDataset containing the loaded and processed data.
    """

    dataset = get_hf_dataset(dataset, device, verbose)

    # pop fovs from dataset
    fov = dataset.pop("fov", None)
    assert fov is not None, "Dataset must contain 'fov' key."

    if filters is not None:
        if ("PD" in dataset) and (filters.pd_cfg is not None):
            dataset["PD"] = apply_filter(dataset["PD"], filters.pd_cfg)
        if ("T1" in dataset) and (filters.t1_cfg is not None):
            dataset["T1"] = apply_filter(dataset["T1"], filters.t1_cfg)
        if ("T2s" in dataset) and (filters.t2s_cfg is not None):
            dataset["T2s"] = apply_filter(dataset["T2s"], filters.t2s_cfg)
        if ("T2" in dataset) and (filters.t2_cfg is not None):
            dataset["T2"] = apply_filter(dataset["T2"], filters.t2_cfg)
        if ("B0" in dataset) and (filters.b0_cfg is not None):
            dataset["B0"] = apply_filter(dataset["B0"], filters.b0_cfg)

    # spatial resize if desired
    if im_size is not None:
        im_size_in = dataset["PD"].shape
        assert len(im_size_in) == len(
            im_size
        ), f"Mismatch in ndim of data ({len(im_size_in)}) and im_size arg ({len(im_size)})."
        if verbose:
            print(
                f"Resizing dataset from {im_size_in} to {im_size} using `{resize_method}` method."
            )
        for key in dataset.keys():
            dataset[key] = spatial_resize(
                dataset[key],
                im_size,
                method=resize_method,
                window=window_method,
            )

    return QuantitativeDataset(
        fov=fov,
        PD=dataset["PD"],
        T2s=dataset.get("T2s", None),
        T2=dataset.get("T2", None),
        T1=dataset.get("T1", None),
        B1=dataset.get("B1", None),
        B0=dataset.get("B0", None),
        mps=dataset.get("mps", None),
        mask=dataset.get("mask", None),
    )
