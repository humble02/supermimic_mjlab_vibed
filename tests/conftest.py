from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import MJCF_PATH, default_joint_pos_list


@pytest.fixture(scope="session")
def motion_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("motions")
    # Lengths resample to 51 and 101: exactly the terminal-only-bin edge case.
    for index, length in enumerate((31, 61)):
        t = np.arange(length) / 30.0
        data = np.zeros((length, 36))
        data[:, 0] = 0.3 * t
        data[:, 2] = 0.78
        yaw = 1.2 + 0.4 * t
        data[:, 5], data[:, 6] = np.sin(yaw / 2), np.cos(yaw / 2)
        data[:, 7:] = np.array(default_joint_pos_list()) + 0.015 * np.sin(t[:, None] * 2)
        np.savetxt(directory / f"clip{index}.csv", data, delimiter=",")
    return sorted(str(p) for p in directory.glob("*.csv"))


@pytest.fixture(scope="session")
def cpu_motion(motion_files):
    if not Path(MJCF_PATH).is_file():
        pytest.skip("Prepare the G1 asset with tools/download_data.sh")
    kin = G1Kinematics("cpu")
    return MotionLib(motion_files, kin, "cpu")


@pytest.fixture(scope="session")
def gpu_components(motion_files):
    if not torch.cuda.is_available() or not Path(MJCF_PATH).is_file():
        pytest.skip("CUDA and the prepared G1 asset are required")
    device = "cuda:0"
    kin = G1Kinematics(device)
    motion = MotionLib(motion_files, kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    return kin, motion, mdp
