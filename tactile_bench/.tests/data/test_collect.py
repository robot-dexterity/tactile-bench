"""Unit tests for tactile_bench.data.collect's target generation helpers.

Run from the repository root with:
    uv run pytest tactile_bench/data/tests/test_collect.py -v
"""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tactile_bench.data import collect


@pytest.fixture
def globals_cfg():
    settings_path = Path(__file__).parents[2] / "cfg" / "settings.yaml"
    globals_config = OmegaConf.load(settings_path)["globals"]
    globals_config.REFERENCE = None
    return globals_config


class RecordingRobot:
    def __init__(self):
        self.joint_angles = [10, 20, 30]
        self.linear_moves = []
        self.joint_moves = []
        self.closed = False

    def move_linear(self, pose):
        self.linear_moves.append(tuple(pose))

    def move_joints(self, joint_angles):
        self.joint_moves.append(tuple(joint_angles))

    def close(self):
        self.closed = True


class RecordingSensor:
    def __init__(self):
        self.events = []

    @contextmanager
    def capture_images(self, image_file):
        self.events.append(("capture_begin", image_file))
        yield
        self.events.append(("capture_end", image_file))

def make_target(globals_cfg, image, object_class, pose, shear, object_pose):
    return {
        globals_cfg.IMAGE: image,
        globals_cfg.OBJECT_CLASS: object_class,
        **dict(zip(globals_cfg.POSE, pose)),
        **dict(zip(globals_cfg.SHEAR, shear)),
        **dict(zip(globals_cfg.OBJECT_POSE, object_pose)),
    }


def test_setup_targets_builds_expected_rows_and_writes_csv(tmp_path, globals_cfg, monkeypatch):
    poses = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=float)
    shears = np.array([[21, 22, 23, 24, 25, 26], [27, 28, 29, 30, 31, 32]], dtype=float)
    generated = iter((poses, shears, poses + 100, shears + 100))
    monkeypatch.setattr(collect, "sample_poses", lambda *_: next(generated))
    config = OmegaConf.create(
        {
            "object_poses": {"first": [0, 1, 2, 3, 4, 5], "second": [10, 11, 12, 13, 14, 15]},
            "pose_lims": [[-1] * 6, [1] * 6],
            "shear_lims": [[-2] * 6, [2] * 6],
        }
    )

    targets = collect.setup_targets(2, config, globals_cfg, str(tmp_path))

    expected_columns = [
        globals_cfg.IMAGE,
        globals_cfg.OBJECT_CLASS,
        *globals_cfg.POSE,
        *globals_cfg.SHEAR,
        *globals_cfg.OBJECT_POSE,
    ]
    assert list(targets.columns) == expected_columns
    assert targets[globals_cfg.IMAGE].tolist() == [
        f"{globals_cfg.IMAGE}/1.png",
        f"{globals_cfg.IMAGE}/2.png",
        f"{globals_cfg.IMAGE}/3.png",
        f"{globals_cfg.IMAGE}/4.png",
    ]
    assert targets[globals_cfg.OBJECT_CLASS].tolist() == ["first", "first", "second", "second"]
    np.testing.assert_allclose(targets.loc[0, list(globals_cfg.POSE)].astype(float), poses[0])
    np.testing.assert_allclose(targets.loc[3, list(globals_cfg.SHEAR)].astype(float), shears[1] + 100)
    np.testing.assert_allclose(targets.loc[2, list(globals_cfg.OBJECT_POSE)].astype(float), config.object_poses.second)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "targets.csv"), targets, check_dtype=False)


def test_run_collect_loop_moves_in_expected_order_and_processes_image(globals_cfg, monkeypatch):
    robot = RecordingRobot()
    sensor = RecordingSensor()
    pose = np.array([1, 2, 3, 4, 5, 6])
    shear = np.array([10, 20, 30, 40, 50, 60])
    object_pose = np.array([100, 200, 300, 400, 500, 600])
    targets = pd.DataFrame(
        [make_target(globals_cfg, "sample.png", "object-a", pose, shear, object_pose)]
    )
    monkeypatch.setattr(collect, "sleep", lambda _: None)

    collect.run_collect_loop(robot, sensor, targets, "output", globals_cfg)

    clearance = np.array([0, 0, -10, 0, 0, 0])
    expected_linear_moves = [
        (0, 0, -50, 0, 0, 0),
        tuple(object_pose + clearance),
        tuple(object_pose + pose + shear + clearance),
        tuple(object_pose + pose + shear),
        tuple(object_pose + pose),
        tuple(object_pose + pose + clearance),
    ]
    assert robot.linear_moves == expected_linear_moves
    assert robot.joint_moves == [
        (10, 20, 0),
        (10, 20, 30),
        (10, 20, 30),
        (10, 20, 0),
    ]
    assert sensor.events == [
        ("capture_begin", "output/sample.png"),
        ("capture_end", "output/sample.png"),
    ]
