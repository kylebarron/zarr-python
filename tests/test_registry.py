import pytest

from zarr.abc.read_backend import ReadBackend
from zarr.core.config import BadConfigError, config
from zarr.registry import get_read_backend_class, register_read_backend


class _StubReadBackend(ReadBackend):
    def can_read(self, store_path, metadata, indexer, prototype, config) -> bool:
        return False

    def read(self, store_path, metadata, indexer, prototype) -> None:
        raise AssertionError("not called in this test")

    async def read_async(self, store_path, metadata, indexer, prototype) -> None:
        raise AssertionError("not called in this test")


def test_get_read_backend_class_none_by_default() -> None:
    assert get_read_backend_class() is None


def test_get_read_backend_class_returns_registered_class() -> None:
    register_read_backend(_StubReadBackend, qualname="stub")
    with config.set({"read_backend": "stub"}):
        assert get_read_backend_class() is _StubReadBackend


def test_get_read_backend_class_unknown_key_raises() -> None:
    with config.set({"read_backend": "does-not-exist"}), pytest.raises(BadConfigError):
        get_read_backend_class()
