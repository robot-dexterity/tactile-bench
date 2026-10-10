"""Robot and tactile-sensor setup shared by data and interactive tools."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import sleep
from typing import Iterator

import cv2
import numpy as np
from cri.controller import Controller
from cri.robot import SyncRobot
from cri.sim.sim_controller import SimController
from omegaconf import DictConfig

from tactile_bench.data.utils.image_transforms import apply, normalise_depth
from tactile_bench.data.utils.utils import merge_configs


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
        """Capture images around the movement performed inside this context."""
        mode = self.sensor_params.get("capture", "final")
        if mode != "final":
            raise ValueError(f"sensor.capture must be 'final', got {mode!r}")

        with self.capture_final(outfile):
            yield  # Robot movement occurs here.

    @contextmanager
    def capture_final(self, outfile):
        yield  # Robot movement occurs before the final image is captured.
        sleep(self.settle)
        self.process(outfile)

    def close(self):
        """Release resources owned by the sensor."""


class SimSensor(BaseSensor):
    def __init__(self, sensor_params=None, embodiment=None):
        super().__init__(sensor_params)
        self.embodiment = embodiment

    def read(self):
        return self.embodiment.get_tactile_observation()


class RealSensor(BaseSensor):
    def __init__(self, sensor_params=None):
        super().__init__(sensor_params)
        self.cam = cv2.VideoCapture(self.sensor_params.get("source", 0))
        self.cam.set(cv2.CAP_PROP_EXPOSURE, self.sensor_params.get("exposure", -7))
        for _ in range(self.sensor_params.get("warmup", 5)):
            self.cam.read()

    def read(self):
        _, img = self.cam.read()
        return img

    def close(self):
        self.cam.release()


class ReplaySensor(BaseSensor):
    def process(self, outfile):
        return cv2.imread(outfile, cv2.IMREAD_UNCHANGED)


@contextmanager
def open_embodiment(cfg: DictConfig) -> Iterator[tuple[SyncRobot, BaseSensor]]:
    """Create, configure, and close the configured robot and sensor."""
    capture = cfg.sensor.get("capture", "final")
    if capture != "final":
        raise ValueError(f"sensor.capture must be 'final', got {capture!r}")

    with ExitStack() as stack:
        if cfg.backend.name == "real":
            controller = Controller[cfg.collect.robot]()
            stack.callback(controller.close)
            robot = SyncRobot(controller)
            robot.coord_frame = cfg.collect.real.work_frame
            robot.tcp = cfg.collect.real.tcp_pose
            robot.speed = cfg.collect.real.get("speed", 10)
            sensor = RealSensor(cfg.sensor)
        elif cfg.backend.name == "sim":
            from tactile_sim.pybullet_env import pybullet_env
            sensor_params = merge_configs(cfg.sensor, cfg.sensor.sim)
            env_params = merge_configs(cfg.backend.env, cfg.collect.sim, cfg.sensor.sim)
            depth = env_params.pop("depth", None) or {}  # Translate storage settings for Tactile Sim.
            env_params["depth_dtype"] = "float32" if depth.get("npz") else depth.get("png", "uint8")
            env = pybullet_env(**env_params)
            stack.callback(env.close)
            robot = SyncRobot(SimController(env.arm))
            robot.coord_frame = cfg.collect.sim.work_frame
            robot.tcp = cfg.collect.sim.tcp_pose
            sensor = SimSensor(sensor_params, env)
        else:
            raise ValueError("backend must be one of: sim, real")

        if cfg.sensor.get("replay"):
            sensor.close()
            sensor = ReplaySensor(cfg.sensor)

        stack.callback(sensor.close)
        robot.name, sensor.name = cfg.collect.robot, cfg.sensor.name
        yield robot, sensor  # The caller operates the configured resources here.
