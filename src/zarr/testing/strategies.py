from typing import Any

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings  # noqa: F401
from hypothesis.strategies import SearchStrategy

import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.array import Array
from zarr.core.common import ZarrFormat
from zarr.core.sync import sync
from zarr.storage import MemoryStore, StoreLike
from zarr.storage._common import _dereference_path

# Copied from Xarray
_attr_keys = st.text(st.characters(), min_size=1)
_attr_values = st.recursive(
    st.none() | st.booleans() | st.text(st.characters(), max_size=5),
    lambda children: st.lists(children) | st.dictionaries(_attr_keys, children),
    max_leaves=3,
)


def v3_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        # | npst.byte_string_dtypes(endianness="=")
        # | npst.unicode_string_dtypes()
        # | npst.datetime64_dtypes()
        # | npst.timedelta64_dtypes()
    )


def v2_dtypes() -> st.SearchStrategy[np.dtype]:
    return (
        npst.boolean_dtypes()
        | npst.integer_dtypes(endianness="=")
        | npst.unsigned_integer_dtypes(endianness="=")
        | npst.floating_dtypes(endianness="=")
        | npst.complex_number_dtypes(endianness="=")
        | npst.byte_string_dtypes(endianness="=")
        | npst.unicode_string_dtypes(endianness="=")
        | npst.datetime64_dtypes(endianness="=")
        # | npst.timedelta64_dtypes()
    )


# From https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#node-names
# 1. must not be the empty string ("")
# 2. must not include the character "/"
# 3. must not be a string composed only of period characters, e.g. "." or ".."
# 4. must not start with the reserved prefix "__"
zarr_key_chars = st.sampled_from(
    ".-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
)
node_names = st.text(zarr_key_chars, min_size=1).filter(
    lambda t: t not in (".", "..") and not t.startswith("__")
)
array_names = node_names
attrs = st.none() | st.dictionaries(_attr_keys, _attr_values)
keys = st.lists(node_names, min_size=1).map("/".join)
paths = st.just("/") | keys
# st.builds will only call a new store constructor for different keyword arguments
# i.e. stores.examples() will always return the same object per Store class.
# So we map a clear to reset the store.
stores = st.builds(MemoryStore, st.just({})).map(lambda x: sync(x.clear()))
compressors = st.sampled_from([None, "default"])
zarr_formats: st.SearchStrategy[ZarrFormat] = st.sampled_from([2, 3])
array_shapes = npst.array_shapes(max_dims=4, min_side=0)


@st.composite  # type: ignore[misc]
def numpy_arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    zarr_formats: st.SearchStrategy[ZarrFormat] = zarr_formats,
) -> Any:
    """
    Generate numpy arrays that can be saved in the provided Zarr format.
    """
    zarr_format = draw(zarr_formats)
    return draw(npst.arrays(dtype=v3_dtypes() if zarr_format == 3 else v2_dtypes(), shape=shapes))


@st.composite  # type: ignore[misc]
def np_array_and_chunks(
    draw: st.DrawFn, *, arrays: st.SearchStrategy[np.ndarray] = numpy_arrays
) -> tuple[np.ndarray, tuple[int, ...]]:  # type: ignore[type-arg]
    """A hypothesis strategy to generate small sized random arrays.

    Returns: a tuple of the array and a suitable random chunking for it.
    """
    array = draw(arrays)
    # We want this strategy to shrink towards arrays with smaller number of chunks
    # 1. st.integers() shrinks towards smaller values. So we use that to generate number of chunks
    numchunks = draw(
        st.tuples(
            *[st.integers(min_value=0 if size == 0 else 1, max_value=size) for size in array.shape]
        )
    )
    # 2. and now generate the chunks tuple
    chunks = tuple(
        size // nchunks if nchunks > 0 else 0
        for size, nchunks in zip(array.shape, numchunks, strict=True)
    )
    return (array, chunks)


@st.composite  # type: ignore[misc]
def arrays(
    draw: st.DrawFn,
    *,
    shapes: st.SearchStrategy[tuple[int, ...]] = array_shapes,
    compressors: st.SearchStrategy = compressors,
    stores: st.SearchStrategy[StoreLike] = stores,
    paths: st.SearchStrategy[str | None] = paths,
    array_names: st.SearchStrategy = array_names,
    arrays: st.SearchStrategy | None = None,
    attrs: st.SearchStrategy = attrs,
    zarr_formats: st.SearchStrategy = zarr_formats,
) -> Array:
    store = draw(stores)
    path = draw(paths)
    name = draw(array_names)
    attributes = draw(attrs)
    zarr_format = draw(zarr_formats)
    if arrays is None:
        arrays = numpy_arrays(shapes=shapes, zarr_formats=st.just(zarr_format))
    nparray, chunks = draw(np_array_and_chunks(arrays=arrays))
    # test that None works too.
    fill_value = draw(st.one_of([st.none(), npst.from_dtype(nparray.dtype)]))
    # compressor = draw(compressors)

    expected_attrs = {} if attributes is None else attributes

    array_path = _dereference_path(path, name)
    root = zarr.open_group(store, mode="w", zarr_format=zarr_format)

    a = root.create_array(
        array_path,
        shape=nparray.shape,
        chunks=chunks,
        dtype=nparray.dtype,
        attributes=attributes,
        # compressor=compressor,  # FIXME
        fill_value=fill_value,
    )

    assert isinstance(a, Array)
    if a.metadata.zarr_format == 3:
        assert a.fill_value is not None
    assert a.name is not None
    assert isinstance(root[array_path], Array)
    assert nparray.shape == a.shape
    assert chunks == a.chunks
    assert array_path == a.path, (path, name, array_path, a.name, a.path)
    assert a.basename == name, (a.basename, name)
    assert dict(a.attrs) == expected_attrs

    a[:] = nparray

    return a


def is_negative_slice(idx: Any) -> bool:
    return isinstance(idx, slice) and idx.step is not None and idx.step < 0


@st.composite  # type: ignore[misc]
def basic_indices(draw: st.DrawFn, *, shape: tuple[int], **kwargs: Any) -> Any:
    """Basic indices without unsupported negative slices."""
    return draw(
        npst.basic_indices(shape=shape, **kwargs).filter(
            lambda idxr: (
                not (
                    is_negative_slice(idxr)
                    or (isinstance(idxr, tuple) and any(is_negative_slice(idx) for idx in idxr))
                )
            )
        )
    )


def key_ranges(
    keys: SearchStrategy = node_names, max_size: int | None = None
) -> SearchStrategy[list[int]]:
    """
    Function to generate key_ranges strategy for get_partial_values()
    returns list strategy w/ form::

        [(key, (range_start, range_end)),
         (key, (range_start, range_end)),...]
    """
    byte_ranges = st.builds(
        RangeByteRequest,
        start=st.integers(min_value=0, max_value=max_size),
        end=st.integers(min_value=0, max_value=max_size),
    )
    key_tuple = st.tuples(keys, byte_ranges)
    return st.lists(key_tuple, min_size=1, max_size=10)
