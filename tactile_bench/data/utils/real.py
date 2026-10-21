"""Real robot and camera backend, selected by ``backend=real``.

Opens the CRI robot controller named by ``collect.real.robot`` (or ``collect.robot``)
and an OpenCV camera.
"""
import cv2
from cri.controller import Controller
from cri.robot import SyncRobot

from tactile_bench.data.utils.embodiment import BaseSensor, backend_sensor_params


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


def open_embodiment(cfg, stack):
    """Open the configured robot controller and camera."""
    backend_cfg = cfg.collect.real
    controller = Controller[backend_cfg.get("robot") or cfg.collect.get("robot")]()
    stack.callback(controller.close)
    robot = SyncRobot(controller)
    robot.speed = backend_cfg.get("speed", 10)
    return robot, RealSensor(backend_sensor_params(cfg, "real"))
