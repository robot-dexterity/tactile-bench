# Configuration

The scripts use [Hydra](https://hydra.cc/) for their settings. Each script composes its
configuration from the YAML files in this folder, and any setting can be changed on the
command line without editing them.

## Layout

- `settings.yaml`: shared settings. `settings.path` is where data is saved (`temp` by
  default; use `tactile_data` for data to keep), with the random seed, whether to
  overwrite or restore earlier data (`overwrite_on`, `restore_on`) and the device.
  `globals` names the columns of the targets, such as `POSE` and `OBJECT_POSE`.
- `runtime/base.yaml`: the base of every app. It loads `settings.yaml` and runs Hydra in
  place, without output folders.
- `app/`: each script's own settings and defaults, such as `app/collect.yaml` for
  `data/collect.py`. A script names its app config in its `hydra.main` decorator.
- `backend/`: where to collect: `real` (a robot and camera) or `sim` (Tactile Sim), with
  its environment settings.
- `collect/`: a collection setup: the robot, the stimulus and objects, the pose limits
  to sample, and the work frame and tool pose for each backend.
- `sensor/`: a tactile sensor or camera: its source, image crop and processing, with
  settings for each backend.

## Choosing configs

An app's defaults pick one config from each group: `app/collect.yaml` uses `backend=sim`,
`collect=mixed_xy_mg400` and `sensor=simtip_dome`. Choose others by name on the command
line, from the files in [`collect/`](collect/), [`sensor/`](sensor/) and
[`backend/`](backend/):

```sh
uv run python tactile_bench/data/collect.py collect=mixed_xy_mg400 sensor=simtip_dome
```

Add `--cfg job` to print the composed configuration instead of running.

## Configs from another repository

Every app reads `tactile_bench/cfg`. To run one with another repository's configs, name
the folder and app config with Hydra's `-cp` and `-cn`:

```sh
uv run python tactile_bench/data/collect.py -cp $PWD/../<repository>/cfg
```

## Changing settings

Set any setting by its dotted name. Lists use square brackets and dictionaries curly
braces, with no spaces; `+` adds a setting the config does not have, such as sampling
x and y within a disk:

```sh
uv run python tactile_bench/data/collect.py settings.path=tactile_data collect.num_samples=[10,5]
uv run python tactile_bench/data/collect.py +collect.sample_disk_on=true
```

For a change you want to keep, edit the config file instead, so that it is recorded.

### Sample numbers

`collect.data_dirs` names the splits to collect and `collect.num_samples` gives their
sizes, one each: `[train, val]` and `[800, 200]` in `app/collect.yaml`. A collect config
may set its own, which take precedence. With several objects (`collect.object_poses`),
each split has that many samples per object. With `collect.transfer`, the targets are
those of an earlier collection's splits, and `null` takes them all.

## Multirun

`--multirun` (or `-m`) runs a script once for each combination of comma-separated values,
one after another in the same process:

```sh
uv run python tactile_bench/data/collect.py --multirun \
  collect.num_samples=[3],[5] settings.seed=1,2
```

This runs 4 times, once for each combination of `num_samples` and `seed`.

> [!WARNING]
> As the runs share one process, a run that leaves something in a bad state, such as a
> global variable, can break the runs after it.

## For developers

Each app config starts with `# @package _global_`, so the groups it composes sit at the
top of the configuration: code reads `cfg.collect`, `cfg.sensor` and `cfg.settings.path`,
whichever folder their files are in. A script names its app config, relative to itself:

```python
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../cfg", config_name="app/collect", version_base=None)
def main(cfg: DictConfig):
    print(cfg.collect.name)


if __name__ == "__main__":
    main()
```

`version_base=None` takes the defaults of the installed Hydra version; pin a version,
such as `"1.3"`, to keep its behaviour when Hydra is upgraded.
