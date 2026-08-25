"""Composition tests for the app configuration entry points."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CFG_DIR = Path(__file__).parents[2] / "cfg"


def compose_app(name, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CFG_DIR)):
        cfg = compose(config_name=f"app/{name}", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def test_default_collect_app_composes_sim_surface_dataset():
    collect = compose_app("collect")

    assert collect.backend.name == "sim"
    assert collect.collect.name == "surface_zRxy"
    assert collect.sensor.name == "simtip"
    assert collect.collect.experiment == "simtip/surface_zRxy"


def test_default_process_app_composes_tactip_edge_dataset():
    process = compose_app("process")

    assert process.collect.name == "edge_xRz"
    assert process.sensor.name == "tactip"
    assert process.collect.experiment == "tactip/edge_xRz"
    assert process.sensor.images.circle_mask_radius == 150
    assert process.sensor.images.binary_threshold == [31, -20]


def test_collect_paths_follow_data_root_collection_and_sensor():
    cfg = compose_app(
        "collect",
        "settings.path=custom/sim",
        "collect=edge_xzRxyz_ur",
        "sensor=simtip_dome",
    )

    assert cfg.collect.experiment == "simtip/edge_xzRxyz"
    assert f"{cfg.settings.path}/{cfg.collect.experiment}/train" == (
        "custom/sim/simtip/edge_xzRxyz/train"
    )
