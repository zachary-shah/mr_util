import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download

from ..sim import paths
from ..utils import RESIZE_METHODS, WINDOW_METHODS, spatial_filter, spatial_resize


class QuantitativeDataset:
    def __init__(
        self,
        PD: torch.Tensor,
        T2s: Optional[torch.Tensor] = None,
        T2: Optional[torch.Tensor] = None,
        T1: Optional[torch.Tensor] = None,
        B1: Optional[torch.Tensor] = None,
        B0: Optional[torch.Tensor] = None,
        mps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        """
        Set up a quantitative imaging dataset with the provided tensors.

        Expect all relaxation maps to be in seconds, with B0 in Hz.

        TODO: add support for B1 in acquired model
        """

        self.im_size = PD.shape
        self.device = PD.device
        self.ndim = len(self.im_size)
        assert self.ndim in [2, 3], "Quantitative Image dimensions must be 2D or 3D."

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

    def _evaluate_baseline_signal(self, TR: Optional[float] = None) -> torch.Tensor:
        x = self.PD[
            None,
        ]
        if self.T1 is not None and TR is not None:
            x = x * (
                1
                - torch.exp(
                    -TR
                    / self.T1[
                        None,
                    ]
                )
            )
        return x

    def _apply_b0(self, x: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        if self.B0 is not None:
            x = x.to(torch.complex64)
            x = x * torch.exp(
                -1j
                * 2
                * torch.pi
                * self.B0[
                    None,
                ]
                * tt
            )
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

        # add broadcase dims
        for i in range(self.ndim):
            tt = tt[..., None]

        x = self._evaluate_baseline_signal(TR)
        x = x * torch.exp(
            -tt
            / self.T2s[
                None,
            ]
        ).to(x.dtype)
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

        T2pinv = 1 / (self.T2s + self.eps) - 1 / (self.T2 + self.eps)

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
            x[gre_inds] = x[gre_inds] * torch.exp(
                -tt[gre_inds]
                / self.T2s[
                    None,
                ]
            ).to(x.dtype)
            x[gre_inds] = self._apply_b0(x[gre_inds], tt[gre_inds])
        if se_inds.any():
            tt_se = tt[se_inds]
            x[se_inds] = x[se_inds] * torch.exp(
                -(
                    (tt_se - TE).abs()
                    * T2pinv[
                        None,
                    ]
                )
                - (
                    tt_se
                    / self.T2[
                        None,
                    ]
                )
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
            x = (
                x
                * self.mask[
                    None,
                ]
            )

        if apply_maps:
            assert (
                self.mps is not None
            ), "sensitivity maps must be provided to apply maps."
            x = (
                x[:, None]
                * self.mps[
                    None,
                ]
            )

        if float_input:
            x = x.squeeze(0)

        return x

    def __validate_tensor(self, x, name):
        if x is not None:
            assert (
                x.shape[-self.ndim :] == self.im_size
            ), f"Shape mismatch for {name}: expected {self.im_size}, got {x.shape[-self.ndim:]}"
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device)
        return x


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
        PD=dataset["PD"],
        T2s=dataset.get("T2s", None),
        T2=dataset.get("T2", None),
        T1=dataset.get("T1", None),
        B1=dataset.get("B1", None),
        B0=dataset.get("B0", None),
        mps=dataset.get("mps", None),
        mask=dataset.get("mask", None),
    )
