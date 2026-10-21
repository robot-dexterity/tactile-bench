"""The pose collection protocol: above the target, down to it, capture, and move clear."""
from dataclasses import dataclass
from time import sleep

import numpy as np
from omegaconf import DictConfig

CLEARANCE = (0, 0, -10, 0, 0, 0)  # Offset clear of the object, before and after contact.


@dataclass(frozen=True)
class Target:
    """One sample's target and object poses as arrays."""
    pose: np.ndarray
    object_pose: np.ndarray

    @classmethod
    def from_row(cls, row, globals: DictConfig) -> "Target":
        def values(columns): return row[list(columns)].astype(float).to_numpy()
        return cls(values(globals.POSE), values(globals.OBJECT_POSE))


class ArmResets:
    """ Joint-space reset positions of a collection run: home, clear of the work frame with
    the last joint unwound, and the current object's reset, clear above its pose. Moving
    between them by joint moves avoids winding the wrist on linear paths. """

    def __init__(self, robot: object, clearance=CLEARANCE, home_pose=(0, 0, -50, 0, 0, 0), settle: float = 1):
        self.robot, self.clearance, self.home_pose, self.settle = robot, clearance, home_pose, settle
        self.home = self.object = None

    def start(self) -> None:
        """Move home and record it; it is also the reset until the first object's."""
        self.robot.move_linear(self.home_pose)
        self.home = self.object = (*self.robot.joint_angles[:-1], 0)

    def move_to_object(self, target: Target) -> None:
        """Move clear above a new object's pose and record it as the object's reset."""
        self.robot.move_joints(self.object)
        self.robot.move_linear(target.object_pose + self.clearance)
        sleep(self.settle)
        self.object = self.robot.joint_angles

    def return_to_object(self) -> None:
        self.robot.move_joints(self.object)

    def return_home(self) -> None:
        self.robot.move_joints(self.object)
        self.robot.move_joints(self.home)


def collect_sample(robot, sensor, target: Target, outfile: str) -> None:
    """Pose: move above the target pose and down to it, capture images there, then move
    clear, by CLEARANCE."""
    contact = target.object_pose + target.pose

    robot.move_linear(contact + CLEARANCE)
    robot.move_linear(contact)
    with sensor.capture_images(outfile):
        pass  # No movement: the images are of the target pose.
    robot.move_linear(contact + CLEARANCE)
