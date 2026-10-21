"""Process collected data again, as collection does, with the collect app's config, e.g.:
uv run python tactile_bench/data/process.py collect=mixed_xy_mg400 sensor=tactip_dome127
"""
import cv2
import hydra
import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tactile_bench.utils import image_transforms
from tactile_bench.utils.utils import backup_data, config_source, setup_save_dir


def process_images(
    save_dir: str, images_cfg: DictConfig, globals: DictConfig
) -> None:
    """Process images in given directory, transforming and saving them."""

    print(f"Processing images and targets in {save_dir}...")
    targets_df = pd.read_csv(f"{save_dir}/targets.csv")

    for _, row in tqdm(targets_df.iterrows(), total=len(targets_df)):
        img_name = row[globals.IMAGE]
        img_paths = [path for path in [f"{save_dir}/{img_name}"] if os.path.isfile(path)]

        if not img_paths:
            print(f"Warning: {img_name} missing, skipped.")
            targets_df = targets_df[targets_df[globals.IMAGE] != img_name]
            continue

        for img_path in img_paths:
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

    is_ref = targets[cfg.globals.IMAGE].map(
        lambda image: os.path.basename(os.path.splitext(str(image))[0])
                      == str(cfg.globals.REFERENCE)
    ).to_numpy() if cfg.globals.REFERENCE is not None else np.zeros(len(targets), bool)
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


def process_data(save_dir: str, cfg: DictConfig) -> None:
    """Process a collected split: its images (sensor.images), then its partition
    (collect.partition_dirs)."""
    images_cfg = images_settings(cfg)
    if images_cfg:
        backup_data(save_dir, cfg, "images")
        process_images(save_dir, images_cfg, cfg.globals)
    if cfg.collect.get("partition_dirs", False):
        partition_data(save_dir, cfg)


def images_settings(cfg: DictConfig):
    """The image processing settings: the sensor's (sensor.images), with those a rig's collect
    config gives (collect.real.images); none for a sensor without them, such as a simulated one."""
    if not cfg.sensor.get("images"):
        return None
    rig = (cfg.collect.get("real") or {}).get("images") or {}
    return OmegaConf.merge(cfg.sensor.images, rig)


# The collect app's config, unless the config folder has a process app config of its own.
cfg_path, cfg_name = config_source(__file__, ["app/process", "app/collect"])

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig):
    """For each experiment, process images and save with a dump of the parameters and CSV map to target poses."""

    for dir in cfg.collect.data_dirs:
        process_data(f"{cfg.settings.path}/{cfg.collect.experiment}/{dir}", cfg)


if __name__ == "__main__":
    main()
