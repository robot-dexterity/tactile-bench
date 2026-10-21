"""Collection targets: loaded from an earlier collection (collect.transfer), or sampled
within the configured limits, and written to targets.csv."""
import os
from typing import Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig


REFERENCE_CLASS = "reference"  # The class of the reference sample (globals.REFERENCE): the undeformed sensor.


def transfer_path(save_dir: str, collect: DictConfig) -> str:
    """The transferred targets.csv for a split: collect.transfer/<split>/targets.csv."""
    data_dir = save_dir.split(os.sep)[-1] if save_dir else ""
    return f"{collect.transfer}/{data_dir}/targets.csv"


def setup_targets(
    sample_num: Optional[int], collect: DictConfig, globals: DictConfig, save_dir: str = None,
) -> pd.DataFrame:
    """ Loads the targets of an earlier collection's split (collect.transfer), its first
    sample_num rows (all if None), or else generates them, and writes targets.csv. """

    columns = [globals.IMAGE, globals.OBJECT_CLASS, *globals.POSE, *globals.OBJECT_POSE]

    if collect.get("transfer", None):
        print(f"Loading target poses from {transfer_path(save_dir, collect)}")
        targets_df = pd.read_csv(transfer_path(save_dir, collect), nrows=sample_num)
        targets_df = targets_df.reindex(columns=columns, fill_value=0)
    else:
        targets_df = generate_targets(sample_num, collect, globals, columns)
    targets_df[globals.IMAGE] = [f"{os.path.splitext(str(image))[0]}.png"
                                for image in targets_df[globals.IMAGE]]

    targets_df.to_csv(f"{save_dir}/targets.csv", index=False)
    return targets_df


def generate_targets(
    sample_num: int, collect: DictConfig, globals: DictConfig, columns: list[str]) -> pd.DataFrame:
    """ Samples sample_num targets for each object pose within the configured limits, after
    a reference row if globals.REFERENCE is set. """

    if sample_num is None:
        raise ValueError("num_samples may only be null when transfer is set")
    pose_lims = collect.get("pose_lims", [[0]*6, [0]*6])
    object_poses = collect.get("object_poses", {"contact": [0]*6})
    sample_disk_on = collect.get("sample_disk_on", False)
    sort_on = collect.get("sort_on", False)

    rows = []
    if globals.get("REFERENCE", None) is not None:
        rows.append([f"{globals.IMAGE}/{globals.REFERENCE}",
                     REFERENCE_CLASS, *([0.0] * (len(columns) - 2))])

    for ind, (obj_label, obj_pose) in enumerate(object_poses.items()):
        poses = sample_poses(
            pose_lims[0], pose_lims[1], sample_num, sample_disk_on)
        if sort_on:
            poses = poses[poses[:, -1].argsort()]
        for i in range(sample_num):
            rows.append([f"{globals.IMAGE}/{ind*sample_num + i + 1}",
                        obj_label, *poses[i], *obj_pose])
    return pd.DataFrame(rows, columns=columns)


def sample_poses(
    llims: np.ndarray, ulims: np.ndarray, sample_num: int, sample_disk_on: bool
) -> np.ndarray:
    """ Sample poses uniformly or on disk/sphere within the given limits."""

    llims, ulims = np.array(llims, float), np.array(ulims, float)
    poses_mid, poses_max = (ulims + llims) / 2, (ulims - llims) / 2

    samples = [random_linear(sample_num, x_max) for x_max in poses_max]

    if sample_disk_on:
        # Disk sampling for (x,y) pose components
        if poses_max[0] > 0 and poses_max[1] > 0:
            x, y = random_disk(sample_num, 1)
            samples[0], samples[1] = x * poses_max[0], y * poses_max[1]
        # Spherical sampling for (Rx,Ry) pose components
        if poses_max[3] > 0 and poses_max[4] > 0:
            rx, ry = random_spherical(sample_num, 1)
            samples[3], samples[4] = rx * poses_max[3], ry * poses_max[4]

    return np.array(samples).T + poses_mid


def random_spherical(num_samples, phi_max):
    """Return uniform random sample over a spherical cap bounded by polar angle."""
    phi_max = np.radians(phi_max)
    theta = 2 * np.pi * np.random.rand(num_samples)
    phi = np.arccos(1 - np.random.rand(num_samples) * (1 - np.cos(phi_max)))
    Rx = -np.arcsin(np.sin(phi) * np.sin(theta))
    Ry = -np.arctan2(np.sin(phi) * np.cos(theta), np.cos(phi))
    return np.degrees(Rx), np.degrees(Ry)


def random_disk(num_samples, r_max):
    """Return uniform random sample over a 2D circular disk of radius r_max."""
    r = r_max * np.sqrt(np.random.rand(num_samples))
    theta = 2 * np.pi * np.random.rand(num_samples)
    return r * np.cos(theta), r * np.sin(theta)


def random_linear(num_samples, x_max):
    """Return uniform random sample over a 1D segment [-x_max, x_max]."""
    return -x_max + 2 * x_max * np.random.rand(num_samples)
