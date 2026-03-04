from logging import getLogger
import os
from pathlib import Path
import tarfile
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, IO, Tuple, Mapping

from platformdirs import user_cache_dir
from cosmographi_datasets.loaders import load_rubin_throughput
import requests
import tqdm

logger = getLogger(__name__)


def _get_local_root() -> Path:
    """Get the local cache path, first checking for the env var ``COSMOGRAPHI_CACHE_ROOT``,
    then the user cache, then the install directory.

    Returns
    -------
    Path
        The absolute path to the user cache directory as a Path object.
    """

    if cache_root := os.getenv("COSMOGRAPHI_CACHE_ROOT"):
        cache_root = Path(cache_root)
    else:
        cache_root = Path(user_cache_dir("cosmographi")) / "data"

    return cache_root.resolve()


def get_local_root() -> str:  # pragma: no cover
    """Get the local cache path from :func:`_get_local_root` and create it
    if it doesn't exist.

    Returns
    -------
    str
        The absolute path to the user cache directory.
    """

    local_root = _get_local_root()
    local_root.mkdir(exist_ok=True, parents=True)
    return str(local_root)


REMOTE_ROOT = "https://raw.githubusercontent.com/cosmographi/datasets/loader-test/data"


def download_file(url: str, file_wrapper: IO[bytes]) -> IO[bytes]:
    """Download the file from ``url`` and write to ``file_wrapper``.

    Parameters
    ----------
    url : str
        The url of the targe file.
    file : IO[bytes]
        The open file wrapper to write to.

    Returns
    -------
    IO[bytes]
        The file wrapper ``seek``ed back to 0
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8KB

    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=url) as pbar:
        for data in response.iter_content(block_size):
            file_wrapper.write(data)
            pbar.update(len(data))
    file_wrapper.seek(0)
    return file_wrapper


def download_data(relpath: str) -> None:
    """Download a tarred directory of data from REMOTE_ROOT and extract to ``relpath``
    relative to ``LOCAL_ROOT``.

    Parameters
    ----------
    relpath : str
        The relative path to the directory containing the data

    """

    with NamedTemporaryFile(suffix=".tar.gz", delete_on_close=True) as f:
        # we save it to a temp directory and extract to local root, as we expect all extracted paths to be relative to that one

        remote_path = os.path.join(REMOTE_ROOT, relpath) + ".tar.gz"

        f = download_file(
            remote_path,
            f.file,
        )

        with tarfile.open(fileobj=f, mode="r") as t:
            # We extract straight to local root as the file should be relative to the remote root, which has the same structure
            t.extractall(path=get_local_root())


"""Internal class for holding data loaders for given keys/paths

Note that each loader can be retrieved by its key or its path.
"""


class _Registry:
    """Internal class for holding data loaders for given keys/paths
    Note that each loader can be retrieved by its key or its path.
    """

    _loaders = {}

    def get_loader(
        self, key: str | None = None, relpath: str | None = None
    ) -> Tuple[str, Callable[[str], Dict[str, Any]]]:
        if key is None and relpath is None:
            raise ValueError("key and path cannot both be None!")

        for k in [key, relpath]:
            if k is not None:
                try:
                    loader = self._loaders[k]
                except KeyError:
                    raise KeyError(f"{k} is not in the registry!")

        return loader


# Instantiate our (by convention) singleton registry on first import
_registry = _Registry()


def register_loader(key: str, relpath: str, fn: Callable[[str], Mapping[str, Any]]) -> None:
    """Register a loader with the registry.

    Parameters
    ----------
    key : str
        The unique key for the loader.
    relpath : str
        The unique relative path to the data. This is the path to the directory on the local
        or remote filesystem where the data is located, relative to ``LOCAL_ROOT`` or ``REMOTE_ROOT``.
    fn : Callable[[str], Mapping[str, Any]]
        A function that takes the path to the data directory and returns a dictionary (or TypedDict) of data
        to be passed as kwargs to a class's constructor.

    Raises
    ------
    ValueError
        If ``key`` is already assigned.
    ValueError
        If ``path`` is already assigned.
    """

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

for loader in [rubin_throughput_loader]:
    key, relpath, fn = loader
    register_loader(key, relpath, fn)


def load_data(
    key: str | None = None,
    relpath: str | None = None,
) -> Dict[str, Any]:
    """Load data from the registry by key or path or both (in which case path is a local override).
    The function will first look for the data locally. If both ``key`` and ``relpath`` are passed in,
    the function will raise an exception if the data cannot be found at ``path``.
    Otherwise, if the data can't be found locally, the function will attempt to download it from the remote
    directory and save it to LOCAL_PATH + relpath

    Parameters
    ----------
    key : str | None, optional
        The key of the data, by default None
    relpath : str | None, optional
        The relative path to the dataset, by default None

    Returns
    -------
    Dict[str, Any]
        The data as a dictionary of key-value pairs
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
    args = fn(os.path.join(get_local_root(), _relpath))

    return args


class Loadable:
    """Mixin for classes that can be auto-loaded via the registry."""

    """Load data from ``key`` and/or ``path`` from the registry and use to
    instantiate ``cls`` and return.

    Returns
    -------
    cls
        An instance of ``cls``
    """

    @classmethod
    def load(cls, key: str | None = None, path: str | None = None, *args, **kwargs):
        data = load_data(key, path)
        _kwargs = data | kwargs
        return cls(*args, **_kwargs)
