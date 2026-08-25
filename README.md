# Tactile-Bench: Robot benchmarking suite for tactile robotics

[ ![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Environment Setup](https://github.com/robot-dexterity/tactile-bench/actions/workflows/setup_env.yml/badge.svg)
![Scripts](https://github.com/robot-dexterity/tactile-bench/actions/workflows/script-tests.yml/badge.svg)

## Installation

This repo has been developed and tested with Ubuntu 22.04 and python 3.8 and 3.9
We use `uv` to manage the python environment.

Clone the repository and its dependencies to same root folder:
```console
git clone https://github.com/robot-dexterity/tactile-bench-dev
git clone https://github.com/robot-dexterity/common-robot-interface
git clone https://github.com/robot-dexterity/tactile-sim
cd tactile-bench-dev
```

Install the repository:

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

**macOS (Apple Silicon):** `pybullet` (pulled in by `tactile-sim`) currently builds from source,
and recent Apple SDK headers can fail unless a compatibility flag is set. Use:
```sh
./scripts/uv-mac-sim.sh sync
```
Use the wrapper for subsequent commands that may build/install packages, e.g.:
```sh
./scripts/uv-mac-sim.sh run python tactile_bench/data/collect.py
```
If you prefer a one-off command instead of the wrapper:
```sh
CFLAGS='-Dfdopen=fdopen' uv sync
```
If you still get a compiler error, ensure Xcode Command Line Tools are installed and updated.

**Windows:** `pybullet` (pulled in by `tactile-sim`) has no Windows wheel for Python 3.9, so `uv sync`
builds it from source and needs a C++ compiler. If the build fails with *"Microsoft Visual C++ 14.0
or greater is required"*, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(the **Desktop development with C++** workload), reopen the terminal, and re-run `uv sync`. 

Linux installs a prebuilt wheel and need nothing extra.

## Where to start
Windows/Linux: You can also run these from VS Code.

1. Collect some Tactile Sim data under `temp/sim`
```sh
uv run python tactile_bench/data/collect.py
```

2. (Optionally) process that tactile data
```sh
uv run python tactile_bench/data/process.py
```

These import the default configuration files from tactile_bench/cfg/app, which you can also specify from the command line
```sh
uv run python tactile_bench/data/collect.py --config-path ../cfg --config-name app/collect.yaml
```

Configuration, collection, capture, and output details are documented in
[`docs/configs.md`](docs/configs.md),
[`docs/data_collection.md`](docs/data_collection.md),
[`docs/sensor_capture.md`](docs/sensor_capture.md), and
[`docs/dataset_layout.md`](docs/dataset_layout.md).

### Running without a display
The above commands may give you some output like:
```
        cannot connect to X server
```
This means you don't have an X11 server set up - e.g. you're running on a remote machine.

To fix this you can either connect with X11 forwarding enabled, or use `xvfb` as a virtual display.
