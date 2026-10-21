# Tactile-Bench: Robot benchmarking suite for tactile robotics

Tactile-Bench collects tactile data with a real robot and camera or in Tactile Sim,
and processes it.

## Installation

This repo has been developed and tested with Ubuntu 22.04 and Python 3.9
(`pyproject.toml` requires Python 3.9). We use `uv` to manage the python environment.

Clone tactile-bench and its dependencies to the same root folder:
```console
git clone https://github.com/robot-dexterity/tactile-bench
git clone https://github.com/robot-dexterity/common-robot-interface
git clone https://github.com/robot-dexterity/tactile-sim
cd tactile-bench
```

Check that you have `uv` installed:
```sh
which uv
```

if you don't see output like: `/home/user/.local/bin/uv`, then [install ](https://docs.astral.sh/uv/getting-started/installation/)`uv`:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set the environment up with:
```sh
uv sync
```

**Linux:** `pybullet` (pulled in by `tactile-sim`) installs from a prebuilt wheel, so nothing
extra is needed.

**Windows:** `pybullet` has no Windows wheel for Python 3.9, so `uv sync` builds it from source
and needs a C++ compiler. If the build fails with *"Microsoft Visual C++ 14.0 or greater is
required"*, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(the **Desktop development with C++** workload), reopen the terminal, and re-run `uv sync`.

**macOS (Apple Silicon):** `pybullet` builds from source and needs a compatibility flag for recent
Apple SDKs, which `scripts/uv-mac-sim.sh` sets. Use it in place of `uv` for commands that may
install packages:
```sh
./scripts/uv-mac-sim.sh sync
./scripts/uv-mac-sim.sh run python tactile_bench/data/collect.py
```
If the build still fails, install or update the Xcode Command Line Tools.

### Optional dependencies

`uv sync` installs the optional dependency groups `keyboard` (keyboard input for the
joggers) and `sim` (Tactile Sim backend) by default. Skip one with
`uv sync --no-group <group>`, such as `sim` if you only use a real robot.

## Where to start

Windows/Linux: You can also run these from VS Code.

1. Collect some Tactile Sim data under `temp`
```sh
uv run python tactile_bench/data/collect.py backend=sim
```

2. (Optionally) process that tactile data again. Collection already processes each split as
the sensor and collect configs set (real TacTip images become marker images, with the raw
ones backed up); `process.py` reads the collect app's config, so it reprocesses what
`collect.py` collects, with the same overrides:
```sh
uv run python tactile_bench/data/process.py collect=mixed_xy_mg400 sensor=tactip_dome127
```

These import the default configuration files from tactile_bench/cfg/app. Defaults are set in file, or you can specify from the command line
```sh
uv run python tactile_bench/data/collect.py --config-path ../cfg --config-name app/collect.yaml
```

3. Jog the simulated robot and preview its tactile images:

```sh
uv run python tactile_bench/tools/jogger.py backend=sim
```

4. Preview a camera using the dummy robot:

```sh
uv run python tactile_bench/tools/jogger.py \
  backend=real collect=dummy_robot sensor=camera
```

For a real robot, select a collection config with a `real` block, and the robot's
sensor config.

More on each part in its guide:
[`tactile_bench/data/README.md`](tactile_bench/data/README.md),
[`tactile_bench/tools/README.md`](tactile_bench/tools/README.md).

### Running without a display
The above commands may give you some output like:
```
        cannot connect to X server
```
This means you don't have an X11 server set up - e.g. you're running on a remote machine.

To fix this you can either connect with X11 forwarding enabled, or use `xvfb` as a virtual display.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE).

Copyright (C) 2025-2026 Nathan Lepora.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for the
full terms.
