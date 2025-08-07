import torch
from config import TEST_CUDA, TEST_DEVICE_IDX

from mr_util import PROJECT_ROOT
from mr_util.interp import interp1d_complex


def test_interp1d_cubic():
    if TEST_CUDA:
        device = torch.device(TEST_DEVICE_IDX)
    else:
        device = torch.device("cpu")

    data = torch.load(
        PROJECT_ROOT / "tests/test_data/interp_data.pt",
        weights_only=True,
        map_location=device,
    )
    raw = data["raw"]
    interp_gt = data["interp"]
    tt = data["tt"]
    tt_new = data["tt_new"]

    for backend in ["torch", "scipy"]:
        interp_hat = interp1d_complex(
            raw,
            tt,
            tt_new,
            method="cubic",
            backend=backend,
            mag_phase=False,
        )
        err = (interp_hat - interp_gt).norm() / interp_gt.norm()
        assert err < 1e-4, f"Interp NRMSE too high with {backend} backend: {err:.2e}"


def test_interp1d_all():
    if TEST_CUDA:
        device = torch.device(TEST_DEVICE_IDX)
    else:
        device = torch.device("cpu")

    tt = torch.linspace(0, 10, 100, device=device)
    raw = torch.sin(tt)

    tt_new = torch.linspace(0, 10, 200, device=device)
    interp_gt = torch.sin(tt_new)  # Ground truth for linear interpolation

    raw = raw[:, None, None]
    interp_gt = interp_gt[:, None, None]

    for backend, method in zip(
        ["torch", "scipy", "scipy", "scipy"], ["cubic", "cubic", "quadratic", "linear"]
    ):
        interp_hat = interp1d_complex(
            raw,
            tt,
            tt_new,
            method=method,
            backend=backend,
            mag_phase=False,
        )
        err = (interp_hat - interp_gt).norm() / interp_gt.norm()
        assert err < 1e-3, f"Interp NRMSE too high with {backend} backend: {err:.2e}"
