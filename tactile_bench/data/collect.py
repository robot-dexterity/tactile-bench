"""Example CLI call:
uv run python tactile_bench/data/collect.py collect=mixed_xy_mg400 sensor=simtip_dome
"""
from typing import Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from tactile_bench.data import process
from tactile_bench.data.utils.embodiment import backend_module, open_embodiment
from tactile_bench.data.utils.protocol import ArmResets, Target, collect_sample
from tactile_bench.data.utils.targets import REFERENCE_CLASS, setup_targets
from tactile_bench.utils.utils import config_source, setup_save_dir

np.set_printoptions(precision=2, suppress=True)


def run_collect_loop(
    robot: object, sensor: object, targets_df: pd.DataFrame, save_dir: str,
    globals: DictConfig, sort_on: bool = False
) -> None:
    """Collect each target with the protocol, and the reference sample clear of the object;
    sort_on goes straight from sample to sample, without returning to the object's reset."""

    resets = ArmResets(robot)
    resets.start()

    # ==== data collection loop ====
    prev_class_obj = None
    for i, row in targets_df.iterrows():
        target = Target.from_row(row, globals)
        class_obj = row[globals.OBJECT_CLASS]
        image_file = (f"{save_dir}/{row[globals.IMAGE]}" if save_dir else None)

        print(f"{row[globals.IMAGE]} ({i+1}/{len(targets_df)}) [{class_obj}] pose {target.pose}")

        if class_obj != prev_class_obj:  # A new object class has its own reset.
            resets.move_to_object(target)
            prev_class_obj = class_obj

        if class_obj == REFERENCE_CLASS:  # Undeformed: at the object's reset, clear above it.
            with sensor.capture_images(image_file):
                robot.move_linear(target.object_pose + resets.clearance)  # Already there.
        else:
            collect_sample(robot, sensor, target, image_file)

        if not sort_on:
            resets.return_to_object()

    resets.return_home()


def collect_data(save_dir: str, sample_num: Optional[int], cfg: DictConfig) -> None:
    """ Collect data for the requested robot/sensor, save data in specified directory. """

    backend_module(cfg.backend.name)  # Checks the backend, before any output.
    np.random.seed(cfg.settings.get('seed', None))

    setup_save_dir(save_dir, cfg, sub_dirs=cfg.globals.IMAGE)
    targets_df = setup_targets(sample_num, cfg.collect, cfg.globals, save_dir)
    args = (save_dir, cfg.globals, cfg.collect.get("sort_on", False))
    with open_embodiment(cfg) as (robot, sensor):
        run_collect_loop(robot, sensor, targets_df, *args)


cfg_path, cfg_name = config_source(__file__, "app/collect")

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig):
    """ For each pair [data_dirs, num_samples], collect data and save in dirs set in config or CLI """

    data_dirs = cfg.collect.data_dirs
    sample_nums = cfg.collect.get("num_samples") or [None] * len(data_dirs)  # None: transfer all.
    if len(sample_nums) != len(data_dirs):
        raise ValueError("num_samples and data_dirs must be same length")

    for dir, sample_num in zip(data_dirs, sample_nums):
        save_dir = f"{cfg.settings.path}/{cfg.collect.experiment}/{dir}"
        collect_data(save_dir, sample_num, cfg)
        process.process_data(save_dir, cfg)


if __name__ == "__main__":
    main()
