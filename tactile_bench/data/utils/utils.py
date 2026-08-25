import numpy as np
from omegaconf import OmegaConf


def merge_configs(*configs):
    """Resolve and merge OmegaConf mappings into a plain dictionary."""
    merged = {}
    for config in configs:
        if config is not None:
            merged.update(OmegaConf.to_container(config, resolve=True))
    return merged

def random_spherical(num_samples, phi_max):
    """Return uniform random sample over a spherical cap bounded by polar angle."""
    phi_max = np.radians(phi_max)
    theta = 2 * np.pi * np.random.rand(num_samples)
    phi = np.arccos(1 - np.random.rand(num_samples) * (1 - np.cos(phi_max)))
    Rx = -np.arcsin(np.sin(phi) * np.sin(theta))
    Ry = -np.arctan2(np.sin(phi) * np.cos(theta), np.cos(phi))
    return np.degrees(Rx), np.degrees(Ry)

def random_disk(num_samples, r_max):
    """Return uniform random sample over a 2D circular disk of radius r_max."""
    r = r_max * np.sqrt(np.random.rand(num_samples))
    theta = 2 * np.pi * np.random.rand(num_samples)
    return r * np.cos(theta), r * np.sin(theta)

def random_linear(num_samples, x_max):
    """Return uniform random sample over a 1D segment [-x_max, x_max]."""
    return -x_max + 2 * x_max * np.random.rand(num_samples)
