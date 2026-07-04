""" Example CL call
uv run python tactile_bench/data/collect.py collect=edge_xRz_mg400 sensor=simtip_dome
"""
import hydra
import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
from time import sleep

from cri.controller import Controller
from cri.robot import SyncRobot
from cri.sim.sim_controller import SimController
from tactile_sim.pybullet_env import pybullet_env

from tactile_bench.data import process
from tactile_bench.data.utils.sensors import RealSensor, SimSensor
from tactile_bench.data.utils import utils

np.set_printoptions(precision=2, suppress=True)


def run_collect_loop(
    robot: object, sensor: object, targets_df: pd.DataFrame, save_dir: str,
    globals: DictConfig, sort_on: bool = False
) -> None:
    """ Collect tactile data by moving the robot to target poses and saving sensor images. """

    clearance, reset_pose = (0, 0, -10, 0, 0, 0), (0, 0, -50, 0, 0, 0)
    
    robot.move_linear(reset_pose)
    reset_joint_angles = [(*robot.joint_angles[:-1], 0)]

    # ==== data collection loop ====
    prev_class_obj = None
    for i, row in targets_df.iterrows():
        pose = row[globals.POSE].astype(float).values
        shear = row[globals.SHEAR].astype(float).values
        pose_obj = row[globals.OBJECT_POSE].astype(float).values
        class_obj = row[globals.OBJECT_CLASS]
        image_file = (f"{save_dir}/{row[globals.IMAGE]}" if save_dir else None)

        print(f"{i+1}/{len(targets_df)}: [{class_obj}] " 
              "pose {}, shear {}".format(pose, shear))

        # new reset arm pose if new object pose
        if class_obj != prev_class_obj:
            robot.move_joints(reset_joint_angles[-1])
            robot.move_linear(pose_obj + clearance)
            sleep(1)
            reset_joint_angles.append(robot.joint_angles)
            prev_class_obj = class_obj

        # move robot, collect data, move back
        robot.move_linear(pose_obj + pose + shear + clearance)
        robot.move_linear(pose_obj + pose + shear)
        robot.move_linear(pose_obj + pose)
        sensor.process(image_file)
        robot.move_linear(pose_obj + pose + clearance)

        if not sort_on:
            robot.move_joints(reset_joint_angles[-1])

    # reset robot
    robot.move_joints(reset_joint_angles[-1])
    robot.move_joints(reset_joint_angles[0])
    robot.close()


def setup_targets(
    sample_num: int, collect: DictConfig, globals: DictConfig, save_dir: str = None
) -> pd.DataFrame:
    """ Generates a dataframe with target poses used for data collection. """

    columns = [globals.IMAGE, globals.OBJECT_CLASS,
               *globals.POSE, *globals.SHEAR, *globals.OBJECT_POSE]

    if collect.get("transfer", None):
        data_dir = save_dir.split(os.sep)[-1] if save_dir else ""
        print(f"Loading target poses from {collect.transfer}/{data_dir}/targets.csv")
        targets_df = pd.read_csv(f"{collect.transfer}/{data_dir}/targets.csv"
                     ).reindex(columns=columns, fill_value=0) #.head(1 + sample_num)

    else: 
        pose_lims = collect.get("pose_lims", [[0]*6, [0]*6])
        shear_lims = collect.get("shear_lims", [[0]*6, [0]*6])
        object_poses = collect.get("object_poses", {"contact": [0]*6})
        sample_disk_on = collect.get("sample_disk_on", False)
        sort_on = collect.get("sort_on", False)

        rows = []
        if globals.get("REFERENCE", None) is not None:
            rows.append([f"{globals.IMAGE}/{globals.REFERENCE}.png", "reference"])  # empty pose

        for ind, (obj_label, obj_pose) in enumerate(object_poses.items()):
            poses = sample_poses(
                pose_lims[0], pose_lims[1], sample_num, sample_disk_on)
            shears = sample_poses(
                shear_lims[0], shear_lims[1], sample_num, sample_disk_on)
            if sort_on:
                poses = poses[poses[:, -1].argsort()]
            for i in range(sample_num):
                rows.append([f"{globals.IMAGE}/{ind*sample_num + i + 1}.png",
                            obj_label, *poses[i], *shears[i], *obj_pose])
        targets_df = pd.DataFrame(rows, columns=columns)
    
    targets_df.to_csv(f"{save_dir}/targets.csv", index=False)
    return targets_df


def sample_poses(
    llims: np.ndarray, ulims: np.ndarray, sample_num: int, sample_disk_on: bool
) -> np.ndarray:
    """ Sample poses uniformly or on disk/sphere within the given limits."""

    llims, ulims = np.array(llims, float), np.array(ulims, float)
    poses_mid, poses_max = (ulims + llims) / 2, (ulims - llims) / 2

    samples = [utils.random_linear(sample_num, x_max) for x_max in poses_max]

    if sample_disk_on:
        # Disk sampling for (x,y) pose components
        if poses_max[0] > 0 and poses_max[1] > 0:
            x, y = utils.random_disk(sample_num, 1)
            samples[0], samples[1] = x * poses_max[0], y * poses_max[1]
        # Spherical sampling for (Rx,Ry) pose components
        if poses_max[3] > 0 and poses_max[4] > 0:
            rx, ry = utils.random_spherical(sample_num, 1)
            samples[3], samples[4] = rx * poses_max[3], ry * poses_max[4]

    return np.array(samples).T + poses_mid


def collect_data(save_dir: str, sample_num: int, cfg: DictConfig) -> None:
    """ Collect data for the requested robot/sensor, save data in specified directory. """

    np.random.seed(cfg.settings.get('seed', None))
    process.setup_save_dir(save_dir, cfg, sub_dirs=cfg.globals.IMAGE)

    if cfg.sensor.get('sim', None): # sim robot
        env = pybullet_env(**{**cfg.collect.sim, **cfg.sensor.sim})
        robot = SyncRobot(SimController(env.arm))
        sensor = SimSensor(cfg.sensor.sim, env)
        robot.coord_frame = cfg.collect.sim.work_frame
        robot.tcp = cfg.collect.sim.tcp_pose 

    else: # real robot
        robot  = SyncRobot(Controller[cfg.collect.robot]())
        sensor = RealSensor(cfg.sensor) 
        robot.coord_frame = cfg.collect.real.work_frame
        robot.tcp = cfg.collect.real.tcp_pose    
        robot.speed = cfg.collect.real.get('speed', 10)
 
    targets_df = setup_targets(sample_num, cfg.collect, cfg.globals, save_dir)

    run_collect_loop(robot, sensor, targets_df, save_dir, cfg.globals, 
                     cfg.collect.get("sort_on", False))


cfg_path, cfg_name = "../cfg", "app/collect"

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig):
    """ For each pair [data_dirs, num_samples], collect data and save in dirs set in config or CLI """

    if len(cfg.collect.sample_nums) != len(cfg.collect.data_dirs):
        raise ValueError(f"sample_nums and data_dirs must be same length")

    for dir, sample_num in zip(cfg.collect.data_dirs, cfg.collect.sample_nums):
        save_dir = f"{cfg.settings.path}/{cfg.collect.experiment}/{dir}"
        collect_data(save_dir, sample_num, cfg)

        if cfg.sensor.get("process_images", False):
            process.backup_data(save_dir, cfg, "images")
            process.process_images(
                save_dir, cfg.sensor.process_images, cfg.globals)

        if cfg.collect.get("partition", False):
            process.partition_data(save_dir, cfg)


if __name__ == "__main__":
    main()
