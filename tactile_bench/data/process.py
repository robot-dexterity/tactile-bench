""" Example CL call
uv run python tactile_bench/data/process.py collect=edge_xRz_mg400 sensor=simtip_dome
"""
import cv2
import hydra
import numpy as np
import os
import pandas as pd
import shutil
import yaml
from omegaconf import DictConfig, OmegaConf
from shutil import rmtree
from tqdm import tqdm

from tactile_bench.data.utils import image_transforms


def process_images(
    save_dir: str, images_cfg: DictConfig, globals: DictConfig
) -> None:
    """Process images in given directory, transforming and saving them."""

    print(f"Processing images and targets in {save_dir}...")
    targets_df = pd.read_csv(f"{save_dir}/targets.csv")

    for _, row in tqdm(targets_df.iterrows(), total=len(targets_df)):
        img_name = row[globals.IMAGE]
        img_path = f"{save_dir}/{img_name}"

        if not os.path.isfile(img_path):
            print(f"Warning: {img_name} missing, skipped.")
            targets_df = targets_df[targets_df[globals.IMAGE] != img_name]
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_new = image_transforms.apply(img, **images_cfg)
        cv2.imwrite(img_path, img_new)

        if images_cfg.get("visualise_on", False):
            preview_cfg = dict(channel_mode="colour", normlz=True,
                               dims=[img_new.shape[1], img_new.shape[0]])
            previews = [image_transforms.apply(frame, **preview_cfg) for frame in (img, img_new)]
            cv2.imshow("Processed Images", cv2.hconcat(previews))
            cv2.waitKey(1)

    targets_df.to_csv(f"{save_dir}/targets.csv", index=False)
    cv2.destroyAllWindows()


def partition_data(save_dir: str, cfg: DictConfig) -> list[str]:
    """Partition data while including reference rows in every output."""

    targets = pd.read_csv(f"{save_dir}/targets.csv")
    dirs = [f"{os.path.dirname(save_dir)}/{p}" for p in cfg.collect.partition_dirs]
    ratios = np.asarray(cfg.collect.get("partition_ratios", [1 - 0.2 * (len(dirs) - 1)] 
                                        + [0.2] * (len(dirs) - 1)), float)
    ratios = np.r_[ratios, 1 - ratios.sum()] \
                if len(ratios) == len(dirs) - 1 else ratios
    if len(ratios) != len(dirs) or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("Invalid collect.partition_ratios")
    ratios /= ratios.sum()

    is_ref = targets[cfg.globals.IMAGE].str.contains(f"(?:^|/){cfg.globals.REFERENCE}\\.png$") \
                if cfg.globals.REFERENCE is not None else np.zeros(len(targets), bool)
    refs = np.flatnonzero(is_ref)
    data = np.random.default_rng(cfg.settings.seed).permutation(np.flatnonzero(~is_ref))
    parts = np.split(data, (np.cumsum(ratios[:-1]) * len(data)).astype(int))

    for out_dir, ids in zip(dirs, parts):
        setup_save_dir(out_dir, cfg)
        ids = np.sort(np.r_[refs, ids])
        part = targets.iloc[ids].copy()
        part[cfg.globals.IMAGE] = f"../{os.path.basename(save_dir)}/" + part[cfg.globals.IMAGE]
        part.to_csv(f"{out_dir}/targets.csv", index=False)

    return dirs


def setup_save_dir(save_dir: str, cfg: DictConfig, sub_dirs: list[str] = []) -> None:
    """Create a dataset directory and write its resolved parameters."""
    
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
            shutil.make_archive(f"{target_file}_bak", 'zip', f"{save_dir}", "targets.csv")
        elif cfg.settings.restore_on:
            print(f"Restoring original targets from {target_file}_bak.zip")
            shutil.unpack_archive(f"{target_file}_bak.zip", f"{save_dir}")


cfg_path, cfg_name = "../cfg", "app/process"

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig):
    """For each experiment, process images and save with a dump of the parameters and CSV map to target poses."""

    for dir in cfg.collect.data_dirs:
        save_dir = f"{cfg.settings.path}/{cfg.collect.experiment}/{dir}"

        if cfg.sensor.get("images", False):
            backup_data(save_dir, cfg, "images")
            process_images(save_dir, cfg.sensor.images, cfg.globals)

        if cfg.collect.get("partition_dirs", False):
            partition_data(save_dir, cfg)


if __name__ == "__main__":
    main()
