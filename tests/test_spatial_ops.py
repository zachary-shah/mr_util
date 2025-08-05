import tempfile
from pathlib import Path

import numpy as np
import torch

from mr_util import PROJECT_ROOT
from mr_util.sim.datasets import SimDataset, load_dataset
from mr_util.sim.paths import set_sim_data_dir
from mr_util.utils import normalize, spatial_filter, spatial_resize

set_sim_data_dir("~/data/")

from config import TEST_CUDA, TEST_DEVICE_IDX


def test_spatial_resize_2d():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")

    data = load_dataset(
        dataset=SimDataset.AXIAL_2D,
        device=device,
        verbose=True,
    )

    x = data.PD

    sizes = [(300, 300), (100, 100)]
    methods = ["nearest", "bilinear", "bicubic", "fourier"]
    for new_size in sizes:
        for method in methods:
            resized = spatial_resize(x, new_size, method=method)
            assert (
                resized.shape == new_size
            ), f"Resized shape {resized.shape} does not match expected {new_size}"
            assert (
                resized.device == device
            ), f"Resized tensor is not on the correct device: {resized.device} vs {device}"


def test_spatial_resize_3d():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")

    data = load_dataset(
        dataset=SimDataset.BRAIN_3D,
        device=device,
        verbose=True,
    )

    size = (220, 220, 50)
    methods = ["nearest", "bilinear", "fourier"]
    x = data.PD
    for method in methods:
        if method == "fourier":
            for window in ["hann", "boxcar", "tukey"]:
                resized = spatial_resize(x, size, method=method, window=window)
        resized = spatial_resize(x, size, method=method)
        assert (
            resized.shape == size
        ), f"Resized shape {resized.shape} does not match expected {size}"
        assert (
            resized.device == device
        ), f"Resized tensor is not on the correct device: {resized.device} vs {device}"


def test_spatial_filter():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")

    data = load_dataset(
        dataset=SimDataset.AXIAL_2D,
        device=device,
        verbose=True,
    )
    x = data.T2s
    x_filt = spatial_filter(
        x,
        x.shape,
        gaussian_filter_size=0.1,
        median_filter_size=3,
    )
    assert (
        x_filt.shape == x.shape
    ), f"Filtered shape {x_filt.shape} does not match original {x.shape}"
    assert x_filt.device == device, f"Filtered tensor is not on the correct device"
    del x_filt

    x_filt_2 = spatial_filter(
        x,
        x.shape,
        gaussian_filter_size=(0.1, 0.01),
    )
    assert (
        x_filt_2.shape == x.shape
    ), f"Filtered shape {x_filt_2.shape} does not match original {x.shape}"
    assert x_filt_2.device == device, f"Filtered tensor is not on the correct device"
    del x_filt_2

    x_filt_3 = spatial_filter(
        x,
        x.shape,
        median_filter_size=(3, 5),
    )
    assert (
        x_filt_3.shape == x.shape
    ), f"Filtered shape {x_filt_3.shape} does not match original {x.shape}"
    assert x_filt_3.device == device, f"Filtered tensor is not on the correct device"
    del x_filt_3

    x_filt_4 = spatial_filter(
        x,
        x.shape,
    )
    assert (
        x_filt_4.shape == x.shape
    ), f"Filtered shape {x_filt_4.shape} does not match original {x.shape}"
    assert x_filt_4.device == device, f"Filtered tensor is not on the correct device"
    del x_filt_4

    im_size = x.shape
    x = x[
        None,
    ].repeat_interleave(3, dim=0)
    x_filt_batched = spatial_filter(
        x,
        im_size,
        gaussian_filter_size=0.1,
        median_filter_size=3,
    )
    assert x_filt_batched.shape == (
        3,
        *im_size,
    ), f"Filtered batched shape {x_filt_batched.shape} does not match expected {(3, *im_size)}"
    assert (
        x_filt_batched.device == device
    ), f"Filtered batched tensor is not on the correct device: {x_filt_batched.device} vs {device}"
