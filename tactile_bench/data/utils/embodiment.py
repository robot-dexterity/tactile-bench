"""Robot and tactile-sensor setup shared by data and interactive tools."""

import csv
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter, sleep, time as real_time
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
        self.timestamps = TimestampSession()
        self.sample_interval = 1.0 / params.get("sample_rate", 30.0)
        self.settle = params.get("settle", 0)
        self.depth_png, self.depth_npz = self._storage_options("depth")
        self.depth_dtype = "float32" if self.depth_npz else self.depth_png

    def read(self):
        raise NotImplementedError

    def _process_frame(self, frame=None):
        return apply(self.read() if frame is None else frame, **self.sensor_params)

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

    def _capture_dir(self, outfile):
        outfile = Path(outfile)
        capture_dir = outfile.with_suffix("")
        capture_dir.mkdir(parents=True, exist_ok=True)
        return capture_dir, outfile.suffix or ".png"

    @contextmanager
    def capture_images(self, outfile):
        """Capture images around the movement performed inside this context."""
        mode = self.sensor_params.get("capture", "final")
        if mode == "stream":
            capture_images = self.capture_stream
        elif mode == "pair":
            capture_images = self.capture_pair
        elif mode in ("final", "stream-final"):
            capture_images = self.capture_final  # Both modes save only the final image.
        else:
            raise ValueError(f"invalid sensor.capture mode: {mode}")

        capture_dir = Path(outfile).with_suffix("")
        with self.timestamps.capture(capture_dir):
            with capture_images(outfile):
                yield  # Robot movement occurs here.

    @contextmanager
    def capture_final(self, outfile):
        yield  # Robot movement occurs before the final image is captured.
        sleep(self.settle)
        self.process(outfile)

    @contextmanager
    def capture_pair(self, outfile):
        capture_dir, ext = self._capture_dir(outfile)
        self._save_timestamped_frame(capture_dir, 0, ext)
        yield  # Robot movement occurs between the paired images.
        sleep(self.settle)
        self._save_timestamped_frame(capture_dir, 1, ext)
        self._save_processed_frame(outfile, self._last_processed_frame)

    def _timestamp_metadata(self):
        return {}

    def _save_frame(self, capture_dir, index, ext, frame=None):
        image = f"{index:04d}{ext}"
        self._last_processed_frame = self._process_frame(frame)
        self._save_processed_frame(capture_dir / image, self._last_processed_frame)
        return image

    def _save_timestamped_frame(self, capture_dir, index, ext, frame=None, metadata=None):
        image = self._save_frame(capture_dir, index, ext, frame)
        metadata = self._timestamp_metadata() if metadata is None else metadata
        self.timestamps.add_frame(image, metadata)

    @contextmanager
    def capture_stream(self, outfile):
        raise ValueError("sensor.capture='stream' requires a streaming sensor")

    def close(self):
        """Release resources owned by the sensor."""


class SimSensor(BaseSensor):
    def __init__(self, sensor_params=None, embodiment=None):
        super().__init__(sensor_params)
        self.embodiment = embodiment

    def read(self):
        return self.embodiment.get_tactile_observation()

    def _timestamp_metadata(self):
        return getattr(self.embodiment, "last_tactile_metadata", {})


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


class LocalStreamingSensor(BaseSensor):
    def __init__(self, sensor):
        super().__init__(sensor.sensor_params)
        self.sensor = sensor
        self.timestamps = sensor.timestamps  # Share one CSV session with the wrapped sensor.
        self._exc = None
        self._frame = self.sensor.read()
        self._record = None
        self._record_lock = Lock()
        self._running = True
        self._thread = Thread(target=self._stream, daemon=True)
        self._thread.start()

    def _stream(self):
        next_sample = perf_counter()
        while self._running:
            try:
                self._frame = self.sensor.read()
            except Exception as exc:
                self._exc = exc
                return

            with self._record_lock:
                if self._record:
                    record_dir, ext, index = self._record
                    metadata = self.sensor._timestamp_metadata()
                    self._save_timestamped_frame(record_dir, index, ext, self._frame, metadata)
                    self._record = (record_dir, ext, index + 1)
            next_sample += self.sample_interval  # Use a fixed schedule to avoid timing drift.
            sleep(max(0, next_sample - perf_counter()))

    def read(self):
        if self._exc is not None:
            raise self._exc
        return self._frame.copy() if hasattr(self._frame, "copy") else self._frame

    @contextmanager
    def capture_stream(self, outfile):
        record_dir, ext = self._capture_dir(outfile)
        with self._record_lock:
            self._record = (record_dir, ext, 0)
        try:
            yield  # Robot movement is recorded by the local streaming thread.
            sleep(self.settle)
        finally:
            with self._record_lock:
                self._record = None
        self.process(outfile)

    def close(self):
        try:
            self._running = False
            self._thread.join()
        finally:
            self.sensor.close()


class ReplaySensor(BaseSensor):
    def process(self, outfile):
        return cv2.imread(outfile, cv2.IMREAD_UNCHANGED)


class TimestampSession:
    FIELDS = ("image", "real_time", "real_elapsed", "sim_elapsed", "sim_steps")

    def __init__(self):
        self.capture_dir = None
        self.real_origin = None
        self.sim_origin = None
        self.rows = []

    @contextmanager
    def capture(self, capture_dir):
        if self.capture_dir is not None:
            raise RuntimeError("a timestamp session is already active")
        self.capture_dir = capture_dir
        self.real_origin = perf_counter()
        try:
            yield  # Image capture appends timestamp rows here.
            self._write()
        finally:
            self.capture_dir = self.real_origin = self.sim_origin = None
            self.rows = []

    def add_frame(self, image, metadata):
        sim_elapsed = sim_steps = ""
        if "sim_time" in metadata and "sim_steps" in metadata:
            current = (float(metadata["sim_time"]), int(metadata["sim_steps"]))
            if self.sim_origin is None:
                self.sim_origin = current
            sim_elapsed = f"{current[0] - self.sim_origin[0]:.6f}"
            sim_steps = current[1] - self.sim_origin[1]
        frame_real_time = metadata.get("real_time", real_time())
        real_elapsed = metadata.get("real_elapsed", perf_counter() - self.real_origin)
        self.rows.append([
            image, f"{float(frame_real_time):.6f}", f"{float(real_elapsed):.6f}",
            sim_elapsed, sim_steps,
        ])

    def _write(self):
        if not self.rows:
            return
        with open(self.capture_dir / "timestamps.csv", "w", newline="") as file:
            csv.writer(file).writerows([self.FIELDS, *self.rows])


@contextmanager
def open_embodiment(cfg: DictConfig) -> Iterator[tuple[SyncRobot, BaseSensor]]:
    """Create, configure, and close the configured robot and sensor."""
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

        if (cfg.sensor.get("capture", "final") in ("stream", "stream-final")
                and not cfg.sensor.get("replay")):
            sensor = LocalStreamingSensor(sensor)

        stack.callback(sensor.close)
        robot.name, sensor.name = cfg.collect.robot, cfg.sensor.name
        yield robot, sensor  # The caller operates the configured resources here.
