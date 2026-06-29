"""Tests for the read-backend seams in Array/AsyncArray.

These exercise the seam wiring with a recording stub backend, so they need no
real backend (e.g. zarrista) installed.
"""

import numpy as np

import zarr
from zarr.abc.read_backend import ReadBackend
from zarr.core.config import config
from zarr.registry import register_read_backend

_CALLS: list[tuple[str, tuple[int, ...]]] = []


class _RecordingBackend(ReadBackend):
    can = True

    def can_read(self, store_path, metadata, indexer, prototype, config) -> bool:
        _CALLS.append(("can_read", indexer.shape))
        return self.can

    def read(self, store_path, metadata, indexer, prototype) -> np.ndarray:
        _CALLS.append(("read", indexer.shape))
        return np.zeros(indexer.shape, dtype="int32")

    async def read_async(self, store_path, metadata, indexer, prototype) -> np.ndarray:
        _CALLS.append(("read_async", indexer.shape))
        return np.zeros(indexer.shape, dtype="int32")


class _CanBackend(_RecordingBackend):
    can = True


class _CannotBackend(_RecordingBackend):
    can = False


register_read_backend(_CanBackend, qualname="recording_can")
register_read_backend(_CannotBackend, qualname="recording_cannot")


def _make_array() -> zarr.Array:
    arr = zarr.create_array({}, shape=(8, 8), chunks=(4, 4), dtype="int32", fill_value=0)
    arr[:] = np.arange(64, dtype="int32").reshape(8, 8)
    return arr


def setup_function() -> None:
    _CALLS.clear()


def test_sync_read_uses_backend_read() -> None:
    arr = _make_array()
    with config.set({"read_backend": "recording_can"}):
        out = arr[0:4, 0:4]
    assert ("read", (4, 4)) in _CALLS
    assert ("read_async", (4, 4)) not in _CALLS  # sync path must not go async
    assert np.array_equal(out, np.zeros((4, 4), dtype="int32"))


async def test_async_read_uses_backend_read_async() -> None:
    arr = _make_array()
    with config.set({"read_backend": "recording_can"}):
        out = await arr._async_array.getitem((slice(0, 4), slice(0, 4)))
    assert ("read_async", (4, 4)) in _CALLS
    assert ("read", (4, 4)) not in _CALLS  # async path must not call sync read
    assert np.array_equal(out, np.zeros((4, 4), dtype="int32"))


def test_sync_falls_back_when_cannot_read() -> None:
    arr = _make_array()
    expected = np.arange(64, dtype="int32").reshape(8, 8)[0:4, 0:4]
    with config.set({"read_backend": "recording_cannot"}):
        out = arr[0:4, 0:4]
    assert ("can_read", (4, 4)) in _CALLS
    assert ("read", (4, 4)) not in _CALLS
    assert np.array_equal(out, expected)  # native values, not backend zeros


async def test_async_falls_back_when_cannot_read() -> None:
    arr = _make_array()
    expected = np.arange(64, dtype="int32").reshape(8, 8)[0:4, 0:4]
    with config.set({"read_backend": "recording_cannot"}):
        out = await arr._async_array.getitem((slice(0, 4), slice(0, 4)))
    assert ("can_read", (4, 4)) in _CALLS
    assert ("read_async", (4, 4)) not in _CALLS
    assert np.array_equal(out, expected)


def test_native_path_when_no_backend() -> None:
    arr = _make_array()
    expected = np.arange(64, dtype="int32").reshape(8, 8)[0:4, 0:4]
    out = arr[0:4, 0:4]
    assert _CALLS == []
    assert np.array_equal(out, expected)
