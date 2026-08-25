"""Tests for tactile image capture and persistence."""

import csv

import cv2
import numpy as np
import pytest

from tactile_bench.data.utils.embodiment import BaseSensor, LocalStreamingSensor


class FakeStreamingSensor(BaseSensor):
    def __init__(self):
        super().__init__({"sample_rate": 1000})
        self.closed = False

    def read(self):
        return np.zeros((4, 5), dtype=np.uint8)

    def close(self):
        self.closed = True


def test_depth_png_and_optional_npz(tmp_path):
    depth = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    sensor = BaseSensor({"depth": {"png": "uint16", "npz": "float16"}})
    sensor.read = lambda: depth
    outfile = tmp_path / "depth.png"

    image = sensor.process(outfile)

    saved = cv2.imread(str(outfile), cv2.IMREAD_UNCHANGED)
    assert image.dtype == saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, np.array([[0, 32767, 65535]], dtype=np.uint16))
    with np.load(outfile.with_suffix(".npz")) as data:
        assert data.files == ["depth"]
        assert data["depth"].dtype == np.float16
        np.testing.assert_array_equal(data["depth"], depth.astype(np.float16))


def test_depth_npz_none_only_saves_png(tmp_path):
    sensor = BaseSensor({"depth": {"png": "uint8", "npz": None}})
    sensor.read = lambda: np.ones((2, 2), dtype=np.float32)
    outfile = tmp_path / "depth.png"

    sensor.process(outfile)

    assert cv2.imread(str(outfile), cv2.IMREAD_UNCHANGED).dtype == np.uint8
    assert not outfile.with_suffix(".npz").exists()


def test_storage_options_are_validated():
    with pytest.raises(ValueError, match="depth.png"):
        BaseSensor({"depth": {"png": "uint12"}})
    with pytest.raises(ValueError, match="depth.png"):
        BaseSensor({"depth": {"png": None, "npz": "float32"}})
    with pytest.raises(ValueError, match="depth.npz"):
        BaseSensor({"depth": {"npz": "uint16"}})


def test_real_timestamp_rows_leave_simulator_fields_blank(tmp_path):
    sensor = BaseSensor()
    with sensor.timestamps.capture(tmp_path):
        sensor.timestamps.add_frame("0000.png", {})

    with open(tmp_path / "timestamps.csv", newline="") as file:
        row = next(csv.DictReader(file))
    assert list(row) == ["image", "real_time", "real_elapsed", "sim_elapsed", "sim_steps"]
    assert row["sim_elapsed"] == ""
    assert row["sim_steps"] == ""


def test_local_streaming_sensor_shares_timestamp_logger():
    inner = FakeStreamingSensor()
    sensor = LocalStreamingSensor(inner)
    try:
        assert sensor.timestamps is inner.timestamps
    finally:
        sensor.close()
