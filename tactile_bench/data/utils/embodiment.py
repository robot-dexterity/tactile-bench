"""Robot and tactile-sensor setup shared by data and interactive tools: one robot/sensor
pair."""

import importlib
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import sleep
from typing import Iterator

import cv2
import numpy as np
from cri.robot import SyncRobot
from omegaconf import DictConfig, OmegaConf

from tactile_bench.utils.image_transforms import apply, normalise_depth


class BaseSensor:
    def _storage_options(self, name):
        config = self.sensor_params.get(name) or {}
        png, npz = config.get("png", "uint8"), config.get("npz")
        if png not in ("uint8", "uint16"):
            raise ValueError(f"sensor.{name}.png must be uint8 or uint16")
        if npz not in (None, "float16", "float32"):
            raise ValueError(f"sensor.{name}.npz must be null, float16, or float32")
        return png, npz

    def __init__(self, sensor_params=None):
        params = self.sensor_params = sensor_params or {}
        self.settle = params.get("settle", 0)
        self.depth_png, self.depth_npz = self._storage_options("depth")
        self.depth_dtype = "float32" if self.depth_npz else self.depth_png

    def read(self):
        raise NotImplementedError

    def _process_frame(self):
        return apply(self.read(), **self.sensor_params)

    def _save_processed_frame(self, outfile, frame):
        outfile = Path(outfile)
        if "depth" not in self.sensor_params:
            cv2.imwrite(str(outfile), frame)
            return frame

        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]

        png_dtype = np.dtype(self.depth_png)
        depth = (normalise_depth(frame)
                 if self.depth_npz or frame.dtype != png_dtype else None)
        if self.depth_npz:
            np.savez_compressed(
                outfile.with_suffix(".npz"), depth=depth.astype(self.depth_npz))
        image = (frame if frame.dtype == png_dtype else
                 (depth * np.iinfo(png_dtype).max).astype(png_dtype))
        cv2.imwrite(str(outfile.with_suffix(".png")), image)
        return image

    def process(self, outfile=None):
        frame = self._process_frame()
        return self._save_processed_frame(outfile, frame) if outfile else frame

    @contextmanager
    def capture_images(self, outfile):
        """Capture the image after the movement performed inside this context."""
        yield  # Robot movement occurs here.
        sleep(self.settle)
        self.process(outfile)

    def close(self):
        """Release resources owned by the sensor."""


class SimSensor(BaseSensor):
    """Sensor read from Tactile Sim's tactile observations."""

    def __init__(self, sensor_params=None, embodiment=None):
        super().__init__(sensor_params)
        self.embodiment = embodiment

    def read(self):
        return self.embodiment.get_tactile_observation()



def merge_configs(*configs):
    """Resolve and merge OmegaConf mappings into a plain dictionary."""
    configs = [OmegaConf.to_container(config, resolve=True)
               if OmegaConf.is_config(config) else config
               for config in configs if config is not None]
    return OmegaConf.to_container(OmegaConf.merge(*configs), resolve=True)


def backend_sensor_params(cfg, backend):
    """Sensor settings for a backend: sensor, sensor.<backend> and collect.<backend>, merged in
    that order."""
    return merge_configs(cfg.sensor, cfg.sensor.get(backend), cfg.collect.get(backend))


def backend_module(name):
    """The real or sim backend module."""
    if name not in ("real", "sim"):
        raise ValueError(f"backend must be real or sim, not {name!r}")
    try:  # Imported here as real.py and sim.py import the sensor classes above.
        return importlib.import_module(f"tactile_bench.data.utils.{name}")
    except ModuleNotFoundError as error:  # Tactile Sim is the optional sim dependency group.
        package = (error.name or name).split(".")[0]
        raise ValueError(f"backend={name} needs the package {package!r}; install its "
                         "dependency group with uv sync (see README.md)") from error


@contextmanager
def open_embodiment(cfg: DictConfig) -> Iterator[tuple[SyncRobot, BaseSensor]]:
    """Create, configure, and close the configured robot and sensor."""
    with ExitStack() as stack:
        backend = cfg.backend.name
        backend_cfg = cfg.collect[backend]

        module = backend_module(backend)  # real or sim.
        robot, sensor = module.open_embodiment(cfg, stack)

        robot.coord_frame = backend_cfg.work_frame
        robot.tcp = backend_cfg.tcp_pose

        stack.callback(sensor.close)
        robot.name = backend_cfg.get("robot") or cfg.collect.get("robot")
        yield robot, sensor  # The caller operates the configured resources here.
