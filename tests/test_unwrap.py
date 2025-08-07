# make something to wrap

import torch
from config import TEST_CUDA, TEST_DEVICE_IDX

from mr_util.sim.datasets import SimDataset, load_dataset
from mr_util.unwrap import torch_unwrap_spatial


def test_unwrap_phase():
    if TEST_CUDA:
        device = torch.device(TEST_DEVICE_IDX)
    else:
        device = torch.device("cpu")

    ds = load_dataset(
        dataset=SimDataset.AXIAL_2D,
        device=device,
        verbose=True,
    )
    B0 = ds.get_maps(["B0"])["B0"]
    gt = B0 * 2 * torch.pi * 20e-3
    phs_wrap = torch.exp(1j * gt)
    phs_unwrap = torch_unwrap_spatial(phs_wrap.angle(), ndim=2)
    nrmse = torch.norm(phs_unwrap - gt) / torch.norm(gt)
    assert nrmse < 1e-4, f"Unwrap NRMSE too high: {nrmse:.2e}"
    assert phs_unwrap.shape == gt.shape, "Unwrapped phase shape mismatch"
    assert phs_unwrap.device == gt.device, "Unwrapped phase device mismatch"

    # batch
    gtb = gt[None, None].expand(2, 3, -1, -1)
    phs_wrapb = torch.exp(1j * gtb)
    phs_unwrapb = torch_unwrap_spatial(phs_wrapb.angle(), ndim=2)
    nrmseb = torch.norm(phs_unwrapb - gtb) / torch.norm(gtb)
    assert nrmseb < 1e-4, f"Unwrap NRMSE too high in batch case: {nrmseb:.2e}"
    assert (
        phs_unwrapb.shape == gtb.shape
    ), "Unwrapped phase shape mismatch in batch case"
    assert (
        phs_unwrapb.device == gtb.device
    ), "Unwrapped phase device mismatch in batch case"
