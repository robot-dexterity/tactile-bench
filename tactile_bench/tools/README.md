# Tools

`jogger.py` moves the robot by keyboard and shows the processed tactile image after
each move.

The jogger uses the same backend, collection and sensor configs as `collect.py`. Its
keyboard input needs the `keyboard` dependency group (`pynput`), which `uv sync`
installs by default. Arrow keys move x/y, Shift + arrows move z/Rx, and Ctrl + arrows
move Ry/Rz. Rx and Ry inputs are disabled for the MG400. Stop with Ctrl+C.

## Where to start

1. Jog the simulated robot and preview its tactile images:

```sh
uv run python tactile_bench/tools/jogger.py backend=sim
```

2. Preview a camera using the dummy robot:

```sh
uv run python tactile_bench/tools/jogger.py \
  backend=real collect=dummy_robot sensor=camera
```

For a real robot, select a collection config with a `real` block, and the robot's
sensor config.
