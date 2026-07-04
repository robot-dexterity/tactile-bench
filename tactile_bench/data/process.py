""" Example CL call
uv run python tactile_bench/data/process.py collect=edge_xRz_mg400 sensor=simtip_dome
"""
import cv2
import hydra
import numpy as np
import os
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from shutil import copy, copytree, rmtree
from tqdm import tqdm

from tactile_bench.data.utils import transform_image
import shutil


def process_images(save_dir: str, process_images: DictConfig, 
                   globals: DictConfig) -> None:
    """Process images in given directory, applying transformations and saving processed images."""

    print(f"Processing images and targets in {save_dir}...")
    targets_df = pd.read_csv(f"{save_dir}/targets.csv")

    for _, row in tqdm(targets_df.iterrows(), total=len(targets_df)):
        img_name = row[globals.IMAGE]
        img_path = f"{save_dir}/{img_name}"

        if not os.path.isfile(img_path):
            print(f"Warning: {img_name} missing, skipped.")
            targets_df = targets_df[targets_df[globals.IMAGE] != img_name]
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_new = transform_image.transform_image(img, **process_images)
        cv2.imwrite(img_path, img_new)

        if process_images.get("visualise_on", False):
            scale = img_new.shape[0] / img.shape[0]
            img = cv2.resize(
                img, (int(img.shape[1] * scale), img_new.shape[0]))
            cv2.imshow("Processed Images", cv2.hconcat([img, img_new]))
            cv2.waitKey(1)

    # params = yaml.safe_load(open(f"{save_dir}/parameters.yaml", "r"))
    # params["sensor"]["process"] = process_images
    # yaml.safe_dump(params, open(f"{save_dir}/parameters.yaml", "w"),
    #                default_flow_style=None)
    targets_df.to_csv(f"{save_dir}/targets.csv", index=False)
    cv2.destroyAllWindows()


def partition_data(save_dir: str, cfg: DictConfig) -> list[str]:
    """Partition data into two e.g. train/val, copying parameters accordingly."""

    np.random.seed(cfg.settings.seed)  # repeatable partitions
    img_id, ref_id = cfg.globals.IMAGE, cfg.globals.REFERENCE, 
    base_dir = os.path.basename(save_dir)
    targets_df = pd.read_csv(f"{save_dir}/targets.csv")

    # set output directories and partition indices
    out_dirs = [f"{os.path.dirname(save_dir)}/{p}" for p in cfg.collect.partition_dirs]
    n = len(out_dirs) - 1
    ratios = cfg.collect.get("partition_ratios", [1 - 0.2 * n] + n * [0.2])  # default 80/20 split

    if len(ratios) == len(out_dirs):
        split_points = np.cumsum(np.asarray(ratios[:-1]) / sum(ratios))
    elif len(ratios) == len(out_dirs) - 1:
        split_points = np.cumsum(ratios)
    else:
        raise ValueError("collect.partition_ratios must be same length as collect.partition_dirs or one fewer")

    num = len(targets_df)
    split_indices = np.clip(np.floor(split_points * num).astype(int), 0, num)
    inds = np.split(np.random.permutation(num), split_indices.tolist())

    ind_ref = np.where(targets_df[img_id].str.contains(f"(^|/){ref_id}\\.png$"))[0]

    for out_dir, ind in zip(out_dirs, inds):
        setup_save_dir(out_dir, cfg)
        # copy(f"{save_dir}/parameters.yaml", f"{out_dir}/parameters.yaml")

        partition_df = targets_df.iloc[np.unique(np.r_[ind_ref, ind])].copy()
        partition_df[img_id] = f"../{base_dir}/" + partition_df[img_id]
        partition_df.to_csv(f"{out_dir}/targets.csv", index=False)

    return out_dirs


def setup_save_dir(save_dir: str, cfg: DictConfig, sub_dirs: list[str] = []) -> None:
    """Create directory for saving models and parameters."""
    
    print(f"Saving to {save_dir}\n Overwrite: {cfg.settings.overwrite_on}")
    if os.path.isdir(save_dir):
        if cfg.settings.overwrite_on:
            print(f"Removing existing directory {save_dir}")
            rmtree(save_dir)
        else:
            raise FileExistsError(f"Save directory {save_dir} exists!")
    os.makedirs(f"{save_dir}")
    for sub_dir in (sub_dirs if isinstance(sub_dirs, list) else [sub_dirs]):
        os.makedirs(f"{save_dir}/{sub_dir}")  # image directory
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(f"{save_dir}/parameters.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=None)


def backup_data(save_dir: str, cfg, data_type: str = None) -> None:
    """Check if backup already exists. If it doesn't, make a backup."""
    image_dir, target_file = f"{save_dir}/{cfg.globals.IMAGE}", f"{save_dir}/targets"
    if data_type == "images":
        if not os.path.isfile(f"{image_dir}_bak.zip"):
            print(f"Backing up original data in {image_dir}_bak.zip")
            shutil.make_archive(f"{image_dir}_bak", 'zip', image_dir)
        elif cfg.settings.restore_on:
            print(f"Restoring original data from {image_dir}_bak.zip")
            shutil.unpack_archive(f"{image_dir}_bak.zip", image_dir)
        if not os.path.isfile(f"{target_file}_bak.zip"):
            print(f"Backing up original targets in {target_file}_bak.zip")
            # copy(f"{target_file}.zip", f"{target_file}_bak.zip")
            shutil.make_archive(f"{target_file}_bak", 'zip', f"{save_dir}", "targets.csv")
        elif cfg.settings.restore_on:
            print(f"Restoring original targets from {target_file}_bak.zip")
            # copy(f"{target_file}_bak.zip", f"{target_file}.zip")
            shutil.unpack_archive(f"{target_file}_bak.zip", f"{save_dir}")


cfg_path, cfg_name = "../cfg", "app/process"

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig):
    """For each experiment, process images and save with a dump of the parameters and CSV map to target poses."""

    for dir in cfg.collect.data_dirs:
        save_dir = f"{cfg.settings.path}/{cfg.collect.experiment}/{dir}"

        if cfg.sensor.get("process_images", False):
            backup_data(save_dir, cfg, "images")
            process_images(save_dir, cfg.sensor.process_images, cfg.globals)

        if cfg.collect.get("partition_dirs", False):
            partition_data(save_dir, cfg)


if __name__ == "__main__":
    main()
