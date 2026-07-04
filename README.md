# Tactile-Bench: Robot benchmarking suite for tactile robotics

[ ![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Environment Setup](https://github.com/robot-dexterity/tactile-bench/actions/workflows/setup_env.yml/badge.svg)
![Scripts](https://github.com/robot-dexterity/tactile-bench/actions/workflows/script-tests.yml/badge.svg)

## Installation

This repo has been developed and tested with Ubuntu 22.04 and python 3.8 and 3.9
We use `uv` to manage the python environment.

Clone the repository and its dependencies to same root folder:

```console
git clone https://github.com/robot-dexterity/tactile-bench
git clone https://github.com/robot-dexterity/common-robot-interface
git clone https://github.com/robot-dexterity/tactile_sim
cd tactile_bench
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

If you use conda or similar environment handler you may need to run:

```sh
uv venv .venv
uv sync
```

## Where to start
Windows/Linux: You can also run these from VS Code.

1. Collect some tactile data and save in temp/tactile_data
```sh
uv run python tactile_bench/data/collect.py

```

2. (Optionally) process that tactile data to binary images
```sh
uv run python tactile_bench/data/process.py
```

These import the default configuration files from tactile_bench/cfg/app, which you can also specify from the command line
```sh
uv run python tactile_bench/data/collect.py --config-path ../cfg --config-name app/collect.yaml
```

### Running without a display
The above commands may give you some output like:
```
        cannot connect to X server
```
This means you don't have an X11 server set up - e.g. you're running on a remote machine.

To fix this you can either connect with X11 forwarding enabled, or use `xvfb` as a virtual display.
