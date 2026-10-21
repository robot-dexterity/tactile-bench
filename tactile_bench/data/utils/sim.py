"""Tactile Sim (PyBullet) backend, selected by ``backend=sim``.

Creates a local Tactile Sim environment from ``backend.env``, ``collect.sim`` and
``sensor.sim``.
"""
from cri.robot import SyncRobot
from cri.sim.sim_controller import SimController
from tactile_sim.pybullet_env import pybullet_env  # The sim dependency group's.

from tactile_bench.data.utils.embodiment import SimSensor, backend_sensor_params, merge_configs


def open_embodiment(cfg, stack):
    """Create the Tactile Sim environment and its simulated robot and sensor."""
    env_params = merge_configs(cfg.backend.env, cfg.collect.sim, cfg.sensor.get("sim"))
    depth = env_params.pop("depth", None) or {}  # Translate storage settings for Tactile Sim.
    env_params["depth_dtype"] = "float32" if depth.get("npz") else depth.get("png", "uint8")
    env = pybullet_env(**env_params)
    stack.callback(env.close)
    return SyncRobot(SimController(env.arm)), SimSensor(backend_sensor_params(cfg, "sim"), env)
