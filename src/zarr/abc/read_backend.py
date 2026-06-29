from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.core.array_spec import ArrayConfig
    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import NDArrayLikeOrScalar
    from zarr.core.indexing import BasicIndexer
    from zarr.core.metadata import ArrayMetadata
    from zarr.storage import StorePath


class ReadBackend(ABC):
    """An optional, opt-in engine that serves whole-selection reads.

    Implementations decide per call (``can_read``) whether they can serve a
    given read; when they decline, zarr-python runs its native read path. The
    backend never owns indexing -- it receives an already-built ``BasicIndexer``.
    """

    @abstractmethod
    def can_read(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        indexer: BasicIndexer,
        prototype: BufferPrototype,
        config: ArrayConfig,
    ) -> bool:
        """Return True iff this backend can serve this exact read."""
        ...

    @abstractmethod
    def read(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        indexer: BasicIndexer,
        prototype: BufferPrototype,
    ) -> NDArrayLikeOrScalar:
        """Serve the read synchronously; only called when ``can_read`` returned True.

        Used by the sync API path (``Array.__getitem__``), bypassing the
        async event-loop bridge so a synchronous backend (e.g. zarrs' sync API)
        is called directly.
        """
        ...

    @abstractmethod
    async def read_async(
        self,
        store_path: StorePath,
        metadata: ArrayMetadata,
        indexer: BasicIndexer,
        prototype: BufferPrototype,
    ) -> NDArrayLikeOrScalar:
        """Serve the read asynchronously; only called when ``can_read`` returned True.

        Used by the async API path (``AsyncArray.getitem``).
        """
        ...
