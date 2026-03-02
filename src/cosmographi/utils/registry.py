import os
import tarfile
from tempfile import NamedTemporaryFile
from typing import Callable, IO

import numpy as np
import pandas as pd
import requests
import tqdm

LOCAL_ROOT = "/tmp/cg_data/data"
REMOTE_ROOT = "https://raw.githubusercontent.com/cosmographi/datasets/loader-test/data"


def download_file(url, file: IO[bytes]):
    """Downloads a file with a tqdm progress bar to file object."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8KB

    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=url) as pbar:
        for data in response.iter_content(block_size):
            file.write(data)
            pbar.update(len(data))
    file.seek(0)


def download_data(relpath: str):
    """Download a tarred directory of data from REMOTE_ROOT and extract to ``relpath``
    relative to ``LOCAL_ROOT``.

    :param relpath: The relative path to the directory containing the data
    :type relpath: Path
    """
    with NamedTemporaryFile(suffix=".tar.gz", delete_on_close=True) as f:
        # we save it to a temp directory and extract to local root, as we expect all extracted paths to be relative to that one

        remote_path = os.path.join(REMOTE_ROOT, relpath) + ".tar.gz"

        download_file(
            remote_path,
            f.file,
        )

        with tarfile.open(f.name, "r") as t:
            # We extract straight to local root as the file should be relative to the remote root, which has the same structure
            t.extractall(path=LOCAL_ROOT)


def load_rubin_throughput(self, abspath: str):
    """Read in data from ```abspath``` and assign as needed to properties of ``self``.
    It's expected that ``self`` would be an instance of ``RubinThroughput``, but this is not required.

    :param abspath: The absolute path to the data on the local filesystem.
    :type abspath: str
    """

    bands = ["u", "g", "r", "i", "z", "y"]
    w_hardware = []
    T_hardware = []
    for b in bands:
        df = pd.read_csv(
            os.path.join(abspath, f"lsst_hardware_{b}.csv"),
            names=["w", "T"],
            comment="#",
        )
        w_hardware.append(df["w"].values)
        T_hardware.append(df["T"].values)
    w_hardware = np.stack(w_hardware)
    T_hardware = np.stack(T_hardware)
    df = pd.read_csv(os.path.join(abspath, "lsst_atmos_10.csv"), names=["w", "T"], comment="#")
    w_atmosphere = df["w"].values
    T_atmosphere = df["T"].values

    self._w = w_hardware
    self._T = T_hardware

    self.bands = bands
    self.w_atmosphere = w_atmosphere
    self.T_atmosphere = T_atmosphere


class _Registry:
    """Internal class for holding data loaders for given keys/paths

    Note that each loader can be retrieved by its key or its path.
    """

    def __init__(self):
        self._loaders = {}

    def get_loader(self, key: str | None = None, relpath: str | None = None):
        if key is None and relpath is None:
            raise ValueError("key and path cannot both be None!")

        for k in [key, relpath]:
            if k is not None:
                try:
                    loader = self._loaders[k]
                except KeyError:
                    raise KeyError(f"{k} is not in the registry!")

        return loader


# Instantiate the singleton registry on first import

_registry = _Registry()


def register_loader(key: str, relpath: str, fn: Callable[[object, str], None]):
    if key in _registry._loaders:
        raise ValueError(f"{key} is already registered!")
    if relpath in _registry._loaders:
        raise ValueError(f"{relpath} is already registered!")

    keys = [k for k in [key, relpath] if k is not None]

    _registry._loaders.update(dict.fromkeys(keys, (relpath, fn)))


# Register the default loaders

rubin_throughput_loader = (
    "rubin_throughput",
    "throughput/rubin/transmissions",
    load_rubin_throughput,
)

some_other_loader = ("some_other_loader_name", "path/to/other/data", lambda x: open(x))

for loader in [rubin_throughput_loader, some_other_loader]:
    key, relpath, fn = loader
    register_loader(key, relpath, fn)


def load_data(
    self,
    key: str | None = None,
    relpath: str | None = None,
):
    """Load data from the registry by key or path or both (in which case path is a local override).
    The function will first look for the data locally. If both ``key`` and ``relpath`` are passed in,
    the function will raise an exception if the data cannot be found at ``path``.
    Otherwise, if the data can't be found locally, the function will attempt to download it from the remote
    directory and save it to LOCAL_PATH + relpath

    :param key: The key of the data, defaults to None
    :type key: str | None, optional
    :param relpath: The relative path to the dataset, defaults to None
    :type relpath: str | None, optional
    """

    _relpath, fn = _registry.get_loader(key, relpath)

    # user provides both, the path may be a custom (local) override, so we'll replace it
    if relpath and key:
        _relpath = relpath

    if _relpath.endswith("/"):
        _relpath = _relpath[:-1]

    # download the data to LOCAL_PATH + relpath if relpath does not exist
    if not os.path.exists(_relpath):
        download_data(_relpath)

    # call the loading function with the path to the local directory where the data is located
    fn(self, os.path.join(LOCAL_ROOT, _relpath))
