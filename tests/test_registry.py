import os

import pytest

from cosmographi.utils.registry import _registry, load_data, download_data, register_loader
from cosmographi.throughput.base import Throughput_wAtmos


def test_registry_startup():
    """Test that the bootstrapped (singleton) registry has the 2 loaders we assigned."""

    # 2 items, 4 keys, since path and key are keys to the dict
    assert len(_registry._loaders) == 4


def test_get_loader():
    """Test the internal get_loader method"""

    loader = _registry.get_loader("rubin_throughput")

    assert loader is not None


def test_download_data(tmp_path, monkeypatch):
    """Test that we can download remote data, extract, and store in expected location"""

    tmp_path = tmp_path / "data"
    tmp_path.mkdir()
    monkeypatch.setattr("cosmographi.utils.registry.LOCAL_ROOT", tmp_path)

    rubin_data_path, _ = _registry.get_loader("rubin_throughput")

    assert not os.path.exists(os.path.join(tmp_path, rubin_data_path))

    download_data(rubin_data_path)

    assert os.path.exists(os.path.join(tmp_path, rubin_data_path))


def test_cannot_register_twice():
    """Test that trying to register the same path twice causes an error"""

    def dummy(p: str):
        return {"foo": 1}

    register_loader("foo", "/tmp", dummy)

    with pytest.raises(ValueError):
        register_loader("foo", "/tmp", dummy)


def test_can_local_override(tmp_path):
    """Test that passing a key and a path to the register function results in a local path override.
    This is useful in situation where the user wants to use a given loader function but on local data.
    """

    def load_fn(fp: str):
        with open(f"{fp}/test_file.txt", "r") as f:
            return {"foo": f.read()}

    class Dummy:
        def __init__(self, foo):
            self.foo = foo

        @classmethod
        def load(cls, key: str, *args, **kwargs):
            data = load_data(key)
            _kwargs = data | kwargs
            return cls(*args, **_kwargs)

    test_dir = tmp_path / "override"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"

    with open(test_file, "w") as f:
        f.write("bar")

    register_loader("test", str(test_dir), load_fn)

    dummy = Dummy.load("test")

    assert dummy.foo == "bar"


def test_load_rubin_throughput(tmp_path, monkeypatch):
    """Test that we can load remote data into a RubinThroughput instance"""
    tmp_path = tmp_path / "data"
    tmp_path.mkdir()
    monkeypatch.setattr("cosmographi.utils.registry.LOCAL_ROOT", tmp_path)

    rubin_throughput = Throughput_wAtmos.load("rubin_throughput")

    assert rubin_throughput.air_mass.value == 1.2
