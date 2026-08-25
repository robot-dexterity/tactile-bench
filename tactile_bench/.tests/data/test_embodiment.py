"""Unit tests for robot and sensor embodiment setup."""

import importlib

from omegaconf import OmegaConf

from tactile_bench.data.utils import embodiment


class RecordingResource:
    def __init__(self, **attributes):
        self.closed = False
        self.__dict__.update(attributes)

    def close(self):
        self.closed = True


def test_open_embodiment_constructs_and_closes_simulated_resources(monkeypatch):
    arm = object()
    env = RecordingResource(arm=arm)
    robot = RecordingResource()
    sim_controller = object()
    sensor = RecordingResource()
    calls = {}

    def fake_pybullet_env(**kwargs):
        calls["environment_kwargs"] = kwargs
        return env

    def fake_sim_controller(received_arm):
        calls["controller_arm"] = received_arm
        return sim_controller

    def fake_sync_robot(controller):
        calls["robot_controller"] = controller
        return robot

    def fake_sim_sensor(sensor_config, received_env):
        calls["sensor_config"] = sensor_config
        calls["sensor_environment"] = received_env
        return sensor

    pybullet_module = importlib.import_module("tactile_sim.pybullet_env")
    monkeypatch.setattr(pybullet_module, "pybullet_env", fake_pybullet_env)
    monkeypatch.setattr(embodiment, "SimController", fake_sim_controller)
    monkeypatch.setattr(embodiment, "SyncRobot", fake_sync_robot)
    monkeypatch.setattr(embodiment, "SimSensor", fake_sim_sensor)
    cfg = OmegaConf.create(
        {
            "backend": {"name": "sim", "env": {"env_option": "env-value"}},
            "collect": {
                "robot": "sim-robot",
                "sim": {
                    "work_frame": [1, 2, 3, 4, 5, 6],
                    "tcp_pose": [7, 8, 9, 10, 11, 12],
                    "arm_option": "arm-value",
                },
            },
            "sensor": {
                "name": "sim-sensor",
                "sim": {
                    "sensor_option": "sensor-value",
                    "depth": {"png": "uint16", "npz": None},
                },
            },
        }
    )

    with embodiment.open_embodiment(cfg) as (returned_robot, returned_sensor):
        assert returned_robot is robot
        assert returned_sensor is sensor
        assert not env.closed
        assert not sensor.closed

    assert env.closed
    assert sensor.closed
    assert calls["controller_arm"] is arm
    assert calls["robot_controller"] is sim_controller
    assert calls["sensor_environment"] is env
    assert calls["sensor_config"] == embodiment.merge_configs(cfg.sensor, cfg.sensor.sim)
    assert calls["environment_kwargs"] == {
        "env_option": "env-value",
        "work_frame": [1, 2, 3, 4, 5, 6],
        "tcp_pose": [7, 8, 9, 10, 11, 12],
        "arm_option": "arm-value",
        "sensor_option": "sensor-value",
        "depth_dtype": "uint16",
    }
    assert robot.coord_frame == cfg.collect.sim.work_frame
    assert robot.tcp == cfg.collect.sim.tcp_pose
    assert robot.name == cfg.collect.robot
    assert sensor.name == cfg.sensor.name


def test_open_embodiment_constructs_and_closes_real_resources(monkeypatch):
    controller = RecordingResource()
    robot = RecordingResource()
    sensor = RecordingResource()
    calls = {}

    def fake_sync_robot(received_controller):
        calls["controller"] = received_controller
        return robot

    def fake_real_sensor(sensor_config):
        calls["sensor_config"] = sensor_config
        return sensor

    monkeypatch.setattr(
        embodiment, "Controller", {"test-controller": lambda: controller}
    )
    monkeypatch.setattr(embodiment, "SyncRobot", fake_sync_robot)
    monkeypatch.setattr(embodiment, "RealSensor", fake_real_sensor)
    cfg = OmegaConf.create(
        {
            "backend": {"name": "real"},
            "collect": {
                "robot": "test-controller",
                "real": {
                    "work_frame": [1, 1, 1, 1, 1, 1],
                    "tcp_pose": [2, 2, 2, 2, 2, 2],
                },
            },
            "sensor": {"name": "real-sensor"},
        }
    )

    with embodiment.open_embodiment(cfg) as (returned_robot, returned_sensor):
        assert returned_robot is robot
        assert returned_sensor is sensor
        assert not controller.closed
        assert not sensor.closed

    assert controller.closed
    assert sensor.closed
    assert calls["controller"] is controller
    assert calls["sensor_config"] == cfg.sensor
    assert robot.coord_frame == cfg.collect.real.work_frame
    assert robot.tcp == cfg.collect.real.tcp_pose
    assert robot.speed == 10
    assert robot.name == cfg.collect.robot
    assert sensor.name == cfg.sensor.name

