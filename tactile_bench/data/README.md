# Data collection and processing

Collection moves a real robot, or one in Tactile Sim, through sampled target poses and
saves the sensor's images with their targets; processing transforms collected images and
splits the datasets. The apps are `collect.py` and `process.py`, configured from
`tactile_bench/cfg`.

## Where to start

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
