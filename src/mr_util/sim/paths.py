from pathlib import Path

SIM_DATA_DIR = "~/data"
HF_REPO = "zachary-shah/mr_sim_data"

SIM_DATASETS = {
    "Brain_3D": "qbrain_3d.pt",
    "Head_3D": "qhead_3d.pt",
    "Axial_2D": "ax_2d.pt",
}


def set_sim_data_dir(path: str):
    global SIM_DATA_DIR
    SIM_DATA_DIR = Path(path).expanduser()
    if not SIM_DATA_DIR.exists():
        SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return SIM_DATA_DIR
