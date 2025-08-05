import tempfile
from pathlib import Path

import numpy as np
import torch

from mr_util.sim import paths
from mr_util.sim.datasets import (
    DatasetFilterConfig,
    MapFilterConfig,
    QuantitativeDataset,
    SimDataset,
    load_dataset,
)
from mr_util.sim.paths import set_sim_data_dir

set_sim_data_dir("~/data/")

from config import TEST_CUDA, TEST_DEVICE_IDX


def test_from_huggingface():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as temp_dir:
        set_sim_data_dir(temp_dir)
        assert Path(paths.SIM_DATA_DIR) == Path(
            temp_dir
        ), "SIM_DATA_DIR was not restored to original value."
        data = load_dataset(
            dataset=SimDataset.BRAIN_3D,
            device=device,
            verbose=True,
        )
        assert isinstance(
            data, QuantitativeDataset
        ), "Loaded data is not a QuantitativeDataset"
        assert any(
            Path(temp_dir).glob("*.pt")
        ), "No .pt files found in test-data directory. Check if the dataset was downloaded correctly."

    # restore path test
    new_path = set_sim_data_dir("~/data/")
    assert (
        paths.SIM_DATA_DIR == new_path
    ), f"SIM_DATA_DIR was not restored to original value: {paths.SIM_DATA_DIR} vs {new_path}"


def test_load_datasets():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")
    set_sim_data_dir("~/data/")

    for dataset in [SimDataset.BRAIN_3D, SimDataset.HEAD_3D, SimDataset.AXIAL_2D]:
        print(f"Testing dataset: {dataset.name}")

        data = load_dataset(
            dataset=dataset,
            device=device,
            verbose=True,
        )

        assert isinstance(
            data, QuantitativeDataset
        ), f"Loaded data is not of type QuantitativeDataset for {dataset.name}"
        assert data.ndim in [
            2,
            3,
        ], f"Data dimensions for {dataset.name} are not 2D or 3D"

        # Check if the data tensors are on the correct device
        assert (
            data.PD.device == device
        ), f"PD tensor for {dataset.name} is not on the correct device"
        assert (
            data.coords.device == device
        ), f"Coords tensor for {dataset.name} is not on the correct device"
        if data.T2s is not None:
            assert (
                data.T2s.device == device
            ), f"T2* tensor for {dataset.name} is not on the correct device"
        if data.T2 is not None:
            assert (
                data.T2.device == device
            ), f"T2 tensor for {dataset.name} is not on the correct device"
        if data.T1 is not None:
            assert (
                data.T1.device == device
            ), f"T1 tensor for {dataset.name} is not on the correct device"
        if data.B0 is not None:
            assert (
                data.B0.device == device
            ), f"B0 tensor for {dataset.name} is not on the correct device"

        print(f"Dataset {dataset.name} passed all checks.")


def test_dataset_preproc_filters():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")
    set_sim_data_dir("~/data/")

    im_size_orig = (168, 134, 146)
    im_size = tuple(int(x * 0.8) for x in im_size_orig)

    data = load_dataset(
        dataset=SimDataset.BRAIN_3D,
        device=device,
        verbose=True,
        im_size=im_size,
        resize_method="bilinear",
        filters=DatasetFilterConfig(
            pd_cfg=MapFilterConfig(clip_min=0.0),
            t2s_cfg=MapFilterConfig(
                clip_min=0.02, clip_max=0.3, gaussian_std=0.05, median_filter_size=3
            ),
            t2_cfg=MapFilterConfig(clip_min=0.0, clip_max=1, gaussian_std=0.01),
            t1_cfg=MapFilterConfig(clip_min=0.0, clip_max=10, gaussian_std=0.01),
            b0_cfg=MapFilterConfig(
                clip_min=-200, clip_max=200, gaussian_std=0.25, median_filter_size=3
            ),
        ),
    )

    assert (
        data.PD.shape == im_size
    ), f"PD shape mismatch: expected {im_size}, got {data.PD.shape}"
    assert (
        data.T2s.shape == im_size
    ), f"T2* shape mismatch: expected {im_size}, got {data.T2s.shape}"
    assert (
        data.T2.shape == im_size
    ), f"T2 shape mismatch: expected {im_size}, got {data.T2.shape}"
    assert (
        data.T1.shape == im_size
    ), f"T1 shape mismatch: expected {im_size}, got {data.T1.shape}"
    assert (
        data.B0.shape == im_size
    ), f"B0 shape mismatch: expected {im_size}, got {data.B0.shape}"


def test_dataset_resize_ops():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")
    set_sim_data_dir("~/data/")

    data = load_dataset(
        dataset=SimDataset.HEAD_3D,
        device=device,
        verbose=True,
        filters=DatasetFilterConfig(
            pd_cfg=MapFilterConfig(clip_min=0.0),
            t2s_cfg=MapFilterConfig(
                clip_min=0.02, clip_max=0.3, gaussian_std=0.05, median_filter_size=3
            ),
            t2_cfg=MapFilterConfig(clip_min=0.0, clip_max=1, gaussian_std=0.01),
            t1_cfg=MapFilterConfig(clip_min=0.0, clip_max=10, gaussian_std=0.01),
            b0_cfg=MapFilterConfig(
                clip_min=-200, clip_max=200, gaussian_std=0.25, median_filter_size=3
            ),
        ),
    )

    # resize fovs
    data.resize_fov((0.19, 0.22, 0.135), corner=(0, -1, 0.08))
    data.resize_fov((0.22, 0.22, 0.135))
    fov = data.fov.cpu().numpy()
    fov_expect = np.array([0.22, 0.22, 0.135])
    im_size_expect = (220, 220, 135)
    assert (
        np.isclose(fov, fov_expect).all().item()
    ), f"FOV mismatch: expected (0.22, 0.22, 0.135), got {data.fov}"
    assert (
        data.im_size == im_size_expect
    ), f"Image size mismatch: expected {im_size_expect}, got {data.im_size}"
    assert (
        data.PD.shape == im_size_expect
    ), f"PD shape mismatch: expected {im_size_expect}, got {data.PD.shape}"

    # resize matrix
    new_size = (256, 256, 256)
    data.resize_matrix(new_size)
    assert (
        data.im_size == new_size
    ), f"Image size mismatch after resize: expected {new_size}, got {data.im_size}"
    assert (
        data.PD.shape == new_size
    ), f"PD shape mismatch after resize: expected {new_size}, got {data.PD.shape}"
    assert data.coords.shape == (
        *new_size,
        3,
    ), f"Coords shape mismatch after resize: expected {(*new_size, 3)}, got {data.coords.shape}"


def test_dataset_evaluate():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")
    set_sim_data_dir("~/data/")

    data = load_dataset(
        dataset=SimDataset.HEAD_3D,
        device=device,
        verbose=True,
        filters=DatasetFilterConfig(
            pd_cfg=MapFilterConfig(clip_min=0.0),
            t2s_cfg=MapFilterConfig(
                clip_min=0.02, clip_max=0.3, gaussian_std=0.05, median_filter_size=3
            ),
            t2_cfg=MapFilterConfig(clip_min=0.0, clip_max=1, gaussian_std=0.01),
            t1_cfg=MapFilterConfig(clip_min=0.0, clip_max=10, gaussian_std=0.01),
            b0_cfg=MapFilterConfig(
                clip_min=-200, clip_max=200, gaussian_std=0.25, median_filter_size=3
            ),
        ),
    )
    # reduce slices
    z_inds = [22, 44, 66, 88, 110]
    data.set_slice((slice(None), slice(None), z_inds))

    # spin echo
    tt = torch.linspace(0.5, 50, 30).to(device).to(torch.float32) / 1000
    TE = 30 / 1000
    TR = 5
    data.evaluate(tt + TE, signal_type="se", TE=TE, TR=TR)
    data.evaluate(tt + TE, signal_type="se", TE=TE, TR=None)
    data.evaluate(tt + TE, signal_type="se", TE=TE, TR=TR, apply_maps=True)
    data.evaluate(tt + TE, signal_type="se", TE=TE, TR=TR, apply_mask=True)
    data.evaluate(
        tt + TE, signal_type="se", TE=TE, TR=TR, apply_mask=True, apply_maps=True
    )

    # gradient echo
    data.evaluate(tt, signal_type="gre", TE=TE, TR=TR)
    data.evaluate(tt, signal_type="gre", TE=TE, TR=None)
    data.evaluate(tt, signal_type="gre", TE=TE, TR=TR, apply_maps=True)
    data.evaluate(tt, signal_type="gre", TE=TE, TR=TR, apply_mask=True)
    data.evaluate(tt, signal_type="gre", TE=TE, TR=TR, apply_mask=True, apply_maps=True)


def test_dataset_coord_ops():
    if TEST_CUDA:
        device_idx = TEST_DEVICE_IDX
        device = torch.device(device_idx)
    else:
        device = torch.device("cpu")
    set_sim_data_dir("~/data/")

    data = load_dataset(
        dataset=SimDataset.HEAD_3D,
        device=device,
        verbose=True,
        filters=DatasetFilterConfig(
            pd_cfg=MapFilterConfig(clip_min=0.0),
            t2s_cfg=MapFilterConfig(
                clip_min=0.02, clip_max=0.3, gaussian_std=0.05, median_filter_size=3
            ),
            t2_cfg=MapFilterConfig(clip_min=0.0, clip_max=1, gaussian_std=0.01),
            t1_cfg=MapFilterConfig(clip_min=0.0, clip_max=10, gaussian_std=0.01),
            b0_cfg=MapFilterConfig(
                clip_min=-200, clip_max=200, gaussian_std=0.25, median_filter_size=3
            ),
        ),
    )
    data.resize_fov((0.19, 0.22, 0.135), corner=(0, -1, 0.08))
    data.resize_fov((0.22, 0.22, 0.135))
    data._update_coordinates(
        iso=torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    )
    data.transpose(dims=(1, 0, 2))
    Nc = data.Nc

    # specific slices
    z_inds = [22, 44, 66, 88, 110]
    targ_size = (data.im_size[0], data.im_size[1], len(z_inds))
    data.set_slice((slice(None), slice(None), z_inds))
    # time-series images
    tt = torch.linspace(0.5, 50, 30).to(device).to(torch.float32) / 1000
    Nt = len(tt)
    TE = 30 / 1000
    tt = tt + TE
    x_t = data.evaluate(tt, TR=None, signal_type="se", TE=TE)

    maps = data.get_maps(return_maps=["B0", "mps", "mask"])
    B0 = maps["B0"]
    mps = maps["mps"]
    mask = maps["mask"]
    coords = data.get_coords()
    fov = data.get_fov()

    assert (
        B0.shape == targ_size
    ), f"B0 shape mismatch: expected {targ_size}, got {B0.shape}"
    assert mps.shape == (
        Nc,
        *targ_size,
    ), f"MPS shape mismatch: expected {(Nc, *targ_size)}, got {mps.shape}"
    assert (
        mask.shape == targ_size
    ), f"Mask shape mismatch: expected {targ_size}, got {mask.shape}"
    assert coords.shape == (
        *targ_size,
        data.ndim,
    ), f"Coords shape mismatch: expected {(*targ_size, data.ndim)}, got {coords.shape}"
    assert x_t.shape == (
        Nt,
        *targ_size,
    ), f"x_t shape mismatch: expected {(Nt, *targ_size)}, got {x_t.shape}"
    assert fov.shape == (3,), f"FOV shape mismatch: expected (3,), got {fov.shape}"
