"""Helpers the apps share, such as their config folder, output folders and backups."""
import os
import shutil
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

PACKAGE = Path(__file__).resolve().parents[1]


def _own_config(names: list[str]) -> tuple[str, str]:
    """tactile_bench/cfg and the first of names it has (else the first)."""
    return str(PACKAGE / "cfg"), next((name for name in names if (PACKAGE / "cfg" / f"{name}.yaml").is_file()),
                                      names[0])




def config_source(app_file: str, config_name) -> tuple[str, str]:
    """The config folder, tactile_bench/cfg, and app config an app (its file, app_file) reads:
    config_name, or the first of a list of them that the folder has, such as evaluation's
    app/evaluate then app/train. Hydra's -cp and -cn override them for a run."""
    return _own_config([config_name] if isinstance(config_name, str) else list(config_name))


def setup_save_dir(save_dir: str, cfg: DictConfig, sub_dirs: list[str] = []) -> None:
    """Create an output directory, with any sub_dirs, and save the resolved config in it as
    parameters.yaml."""
    
    print(f"Saving to {save_dir}\n Overwrite: {cfg.settings.overwrite_on}")
    if os.path.isdir(save_dir):
        if cfg.settings.overwrite_on:
            print(f"Removing existing directory {save_dir}")
            shutil.rmtree(save_dir)
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
        if not os.path.isfile(f"{target_file}_bak.zip"):  # Processing rewrites the targets too.
            print(f"Backing up original targets in {target_file}_bak.zip")
            shutil.make_archive(f"{target_file}_bak", 'zip', f"{save_dir}", "targets.csv")
        elif cfg.settings.restore_on:
            print(f"Restoring original targets from {target_file}_bak.zip")
            shutil.unpack_archive(f"{target_file}_bak.zip", f"{save_dir}")
