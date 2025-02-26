from __future__ import annotations

import asyncio
import contextlib
import pickle
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypedDict

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)
from zarr.core.buffer import Buffer
from zarr.core.buffer.core import BufferPrototype

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Iterable
    from typing import Any

    from obstore import ListResult, ListStream, ObjectMeta, OffsetRange, SuffixRange
    from obstore.store import ObjectStore as _UpstreamObjectStore

    from zarr.core.buffer import BufferPrototype
    from zarr.core.common import BytesLike

__all__ = ["ObjectStore"]

_ALLOWED_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
)


class ObjectStore(Store):
    """A Zarr store that uses obstore for fast read/write from AWS, GCP, Azure.

    Parameters
    ----------
    store : obstore.store.ObjectStore
        An obstore store instance that is set up with the proper credentials.
    read_only : bool
        Whether to open the store in read-only mode.

    Warnings
    --------
    ObjectStore is experimental and subject to API changes without notice. Please
    raise an issue with any comments/concerns about the store.
    """

    store: _UpstreamObjectStore
    """The underlying obstore instance."""

    def __eq__(self, value: object) -> bool:
        from obstore.store import (
            AzureStore,
            GCSStore,
            HTTPStore,
            LocalStore,
            MemoryStore,
            S3Store,
        )

        if not isinstance(value, ObjectStore):
            return False

        if not isinstance(self.store, type(value.store)):
            return False
        if not self.read_only == value.read_only:
            return False

        match value.store:
            case AzureStore():
                assert isinstance(self.store, AzureStore)
                if (
                    (self.store.config != value.store.config)
                    or (self.store.client_options != value.store.client_options)
                    or (self.store.prefix != value.store.prefix)
                    or (self.store.retry_config != value.store.retry_config)
                ):
                    return False
            case GCSStore():
                assert isinstance(self.store, GCSStore)
                if (
                    (self.store.config != value.store.config)
                    or (self.store.client_options != value.store.client_options)
                    or (self.store.prefix != value.store.prefix)
                    or (self.store.retry_config != value.store.retry_config)
                ):
                    return False
            case S3Store():
                assert isinstance(self.store, S3Store)
                if (
                    (self.store.config != value.store.config)
                    or (self.store.client_options != value.store.client_options)
                    or (self.store.prefix != value.store.prefix)
                    or (self.store.retry_config != value.store.retry_config)
                ):
                    return False
            case HTTPStore():
                assert isinstance(self.store, HTTPStore)
                if (
                    (self.store.url != value.store.url)
                    or (self.store.client_options != value.store.client_options)
                    or (self.store.retry_config != value.store.retry_config)
                ):
                    return False
            case LocalStore():
                assert isinstance(self.store, LocalStore)
                if self.store.prefix != value.store.prefix:
                    return False
            case MemoryStore():
                if self.store is not value.store:
                    return False  # Two memory stores can't be equal because we can't pickle memory stores
        return True

    def __init__(self, store: _UpstreamObjectStore, *, read_only: bool = False) -> None:
        import obstore as obs

        if not isinstance(
            store,
            (
                obs.store.AzureStore,
                obs.store.GCSStore,
                obs.store.HTTPStore,
                obs.store.S3Store,
                obs.store.LocalStore,
                obs.store.MemoryStore,
            ),
        ):
            raise TypeError(f"expected ObjectStore class, got {store!r}")
        super().__init__(read_only=read_only)
        self.store = store

    def __str__(self) -> str:
        return f"object://{self.store}"

    def __repr__(self) -> str:
        return f"ObjectStore({self})"

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        state["store"] = pickle.dumps(self.store)
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        state["store"] = pickle.loads(state["store"])
        self.__dict__.update(state)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        # docstring inherited
        import obstore as obs

        try:
            if byte_range is None:
                resp = await obs.get_async(self.store, key)
                return prototype.buffer.from_bytes(await resp.bytes_async())  # type: ignore[arg-type]
            elif isinstance(byte_range, RangeByteRequest):
                bytes = await obs.get_range_async(
                    self.store, key, start=byte_range.start, end=byte_range.end
                )
                return prototype.buffer.from_bytes(bytes)  # type: ignore[arg-type]
            elif isinstance(byte_range, OffsetByteRequest):
                resp = await obs.get_async(
                    self.store, key, options={"range": {"offset": byte_range.offset}}
                )
                return prototype.buffer.from_bytes(await resp.bytes_async())  # type: ignore[arg-type]
            elif isinstance(byte_range, SuffixByteRequest):
                resp = await obs.get_async(
                    self.store, key, options={"range": {"suffix": byte_range.suffix}}
                )
                return prototype.buffer.from_bytes(await resp.bytes_async())  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unexpected byte_range, got {byte_range}")
        except _ALLOWED_EXCEPTIONS:
            return None

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # docstring inherited
        return await _get_partial_values(self.store, prototype=prototype, key_ranges=key_ranges)

    async def exists(self, key: str) -> bool:
        # docstring inherited
        import obstore as obs

        try:
            await obs.head_async(self.store, key)
        except FileNotFoundError:
            return False
        else:
            return True

    @property
    def supports_writes(self) -> bool:
        # docstring inherited
        return True

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()
        if not isinstance(value, Buffer):
            raise TypeError(
                f"ObjectStore.set(): `value` must be a Buffer instance. Got an instance of {type(value)} instead."
            )
        buf = value.to_bytes()
        await obs.put_async(self.store, key, buf)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()
        buf = value.to_bytes()
        with contextlib.suppress(obs.exceptions.AlreadyExistsError):
            await obs.put_async(self.store, key, buf, mode="Create")

    @property
    def supports_deletes(self) -> bool:
        # docstring inherited
        return True

    async def delete(self, key: str) -> None:
        # docstring inherited
        import obstore as obs

        self._check_writable()
        await obs.delete_async(self.store, key)

    @property
    def supports_partial_writes(self) -> bool:
        # docstring inherited
        return False

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        # docstring inherited
        raise NotImplementedError

    @property
    def supports_listing(self) -> bool:
        # docstring inherited
        return True

    def list(self) -> AsyncGenerator[str, None]:
        # docstring inherited
        import obstore as obs

        objects: ListStream[list[ObjectMeta]] = obs.list(self.store)
        return _transform_list(objects)

    def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        import obstore as obs

        objects: ListStream[list[ObjectMeta]] = obs.list(self.store, prefix=prefix)
        return _transform_list(objects)

    def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        # docstring inherited
        import obstore as obs

        coroutine = obs.list_with_delimiter_async(self.store, prefix=prefix)
        return _transform_list_dir(coroutine, prefix)


async def _transform_list(
    list_stream: ListStream[list[ObjectMeta]],
) -> AsyncGenerator[str, None]:
    """
    Transform the result of list into an async generator of paths.
    """
    async for batch in list_stream:
        for item in batch:
            yield item["path"]


async def _transform_list_dir(
    list_result_coroutine: Coroutine[Any, Any, ListResult[list[ObjectMeta]]], prefix: str
) -> AsyncGenerator[str, None]:
    """
    Transform the result of list_with_delimiter into an async generator of paths.
    """
    list_result = await list_result_coroutine

    # We assume that the underlying object-store implementation correctly handles the
    # prefix, so we don't double-check that the returned results actually start with the
    # given prefix.
    prefixes = [obj.lstrip(prefix).lstrip("/") for obj in list_result["common_prefixes"]]
    objects = [obj["path"].lstrip(prefix).lstrip("/") for obj in list_result["objects"]]
    for item in prefixes + objects:
        yield item


class _BoundedRequest(TypedDict):
    """Range request with a known start and end byte.

    These requests can be multiplexed natively on the Rust side with
    `obstore.get_ranges_async`.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    start: int
    """Start byte offset."""

    end: int
    """End byte offset."""


class _OtherRequest(TypedDict):
    """Offset or suffix range requests.

    These requests cannot be concurrent on the Rust side, and each need their own call
    to `obstore.get_async`, passing in the `range` parameter.
    """

    original_request_index: int
    """The positional index in the original key_ranges input"""

    path: str
    """The path to request from."""

    range: OffsetRange | SuffixRange | None
    """The range request type."""


class _Response(TypedDict):
    """A response buffer associated with the original index that it should be restored to."""

    original_request_index: int
    """The positional index in the original key_ranges input"""

    buffer: Buffer
    """The buffer returned from obstore's range request."""


async def _make_bounded_requests(
    store: _UpstreamObjectStore,
    path: str,
    requests: list[_BoundedRequest],
    prototype: BufferPrototype,
) -> list[_Response]:
    """Make all bounded requests for a specific file.

    `obstore.get_ranges_async` allows for making concurrent requests for multiple ranges
    within a single file, and will e.g. merge concurrent requests. This only uses one
    single Python coroutine.
    """
    import obstore as obs

    starts = [r["start"] for r in requests]
    ends = [r["end"] for r in requests]
    responses = await obs.get_ranges_async(store, path=path, starts=starts, ends=ends)

    buffer_responses: list[_Response] = []
    for request, response in zip(requests, responses, strict=True):
        buffer_responses.append(
            {
                "original_request_index": request["original_request_index"],
                "buffer": prototype.buffer.from_bytes(response),  # type: ignore[arg-type]
            }
        )

    return buffer_responses


async def _make_other_request(
    store: _UpstreamObjectStore,
    request: _OtherRequest,
    prototype: BufferPrototype,
) -> list[_Response]:
    """Make suffix or offset requests.

    We return a `list[_Response]` for symmetry with `_make_bounded_requests` so that all
    futures can be gathered together.
    """
    import obstore as obs

    if request["range"] is None:
        resp = await obs.get_async(store, request["path"])
    else:
        resp = await obs.get_async(store, request["path"], options={"range": request["range"]})
    buffer = await resp.bytes_async()
    return [
        {
            "original_request_index": request["original_request_index"],
            "buffer": prototype.buffer.from_bytes(buffer),  # type: ignore[arg-type]
        }
    ]


async def _get_partial_values(
    store: _UpstreamObjectStore,
    prototype: BufferPrototype,
    key_ranges: Iterable[tuple[str, ByteRequest | None]],
) -> list[Buffer | None]:
    """Make multiple range requests.

    ObjectStore has a `get_ranges` method that will additionally merge nearby ranges,
    but it's _per_ file. So we need to split these key_ranges into **per-file** key
    ranges, and then reassemble the results in the original order.

    We separate into different requests:

    - One call to `obstore.get_ranges_async` **per target file**
    - One call to `obstore.get_async` for each other request.
    """
    key_ranges = list(key_ranges)
    per_file_bounded_requests: dict[str, list[_BoundedRequest]] = defaultdict(list)
    other_requests: list[_OtherRequest] = []

    for idx, (path, byte_range) in enumerate(key_ranges):
        if byte_range is None:
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": None,
                }
            )
        elif isinstance(byte_range, RangeByteRequest):
            per_file_bounded_requests[path].append(
                {"original_request_index": idx, "start": byte_range.start, "end": byte_range.end}
            )
        elif isinstance(byte_range, OffsetByteRequest):
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"offset": byte_range.offset},
                }
            )
        elif isinstance(byte_range, SuffixByteRequest):
            other_requests.append(
                {
                    "original_request_index": idx,
                    "path": path,
                    "range": {"suffix": byte_range.suffix},
                }
            )
        else:
            raise ValueError(f"Unsupported range input: {byte_range}")

    futs: list[Coroutine[Any, Any, list[_Response]]] = []
    for path, bounded_ranges in per_file_bounded_requests.items():
        futs.append(_make_bounded_requests(store, path, bounded_ranges, prototype))

    for request in other_requests:
        futs.append(_make_other_request(store, request, prototype))  # noqa: PERF401

    buffers: list[Buffer | None] = [None] * len(key_ranges)

    # TODO: this gather a list of list of Response; not sure if there's a way to
    # unpack these lists inside of an `asyncio.gather`?
    for responses in await asyncio.gather(*futs):
        for resp in responses:
            buffers[resp["original_request_index"]] = resp["buffer"]

    return buffers
