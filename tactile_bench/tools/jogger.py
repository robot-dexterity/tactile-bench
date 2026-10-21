"""Interactively move the robot, showing the processed tactile image after each move.
Arrow keys to move x/y, Shift + arrows z/Rx, Ctrl + arrows Ry/Rz. Stop with Ctrl+C.
For an MG400, Rx and Ry inputs are disabled.

Example:
    uv run python tactile_bench/tools/jogger.py collect=mixed_xy_mg400 sensor=simtip_dome
Camera preview, with the dummy robot:
    uv run python tactile_bench/tools/jogger.py backend=real collect=dummy_robot sensor=camera
"""

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from pynput.keyboard import Key

from tactile_bench.data.utils.embodiment import open_embodiment
from tactile_bench.tools.utils.keyboard import Keyboard
from tactile_bench.utils.utils import config_source

np.set_printoptions(formatter={"float_kind": "{:5.1f}".format})


def pose_increment(keys: frozenset, robot_name: str = "") -> np.ndarray:
    """Map held arrow and modifier keys to a six-axis pose increment."""
    axes = ((4, 5) if keys & {Key.ctrl, Key.ctrl_l, Key.ctrl_r} else
            (2, 3) if keys & {Key.shift, Key.shift_l, Key.shift_r} else (1, 0))
    increment = np.zeros(6)
    increment[list(axes)] = [(Key.down in keys) - (Key.up in keys),
                             (Key.right in keys) - (Key.left in keys)]
    if robot_name.lower() == "mg400":
        increment[[3, 4]] = 0
    return increment


def run_jogger_loop(
    robot: object, sensor: object, num_iterations: int = 10000,
) -> None:
    """Jog from keyboard input."""
    pose = np.array((0, 0, -10, 0, 0, 0), dtype=float)

    try:
        with Keyboard() as keys:
            for i in range(num_iterations):
                pose += pose_increment(keys.get_state(), robot.name)
                robot.move_linear(pose)
                cv2.imshow("tactile image", sensor.process())
                cv2.waitKey(1)  # Let OpenCV draw the window.
                print(f"{i + 1}/{num_iterations}: pose={pose}")
    finally:
        cv2.destroyAllWindows()


cfg_path, cfg_name = config_source(__file__, "app/collect")
@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def main(cfg: DictConfig) -> None:
    """Load the same embodiment as collect.py, then run the jogger."""

    with open_embodiment(cfg) as (robot, sensor):
        run_jogger_loop(robot, sensor)


if __name__ == "__main__":
    main()
