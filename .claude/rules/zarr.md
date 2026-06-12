---
description: 'Zarr 3.x conventions: the v3 store and array API, explicit chunking and compression, group hierarchies, consolidated metadata, and renamed v2 store classes. Apply when writing or reviewing chunked-array storage, zarr stores, or array I/O pipelines.'
paths:
- '**/*.py'
provenance: shared/rules/stack/zarr.md @ 85dde8d
---

You are an expert in chunked array storage with zarr-python 3.

## Principles

1. Chunk shape is a read-pattern decision, never a default to accept silently.
2. A store's layout and metadata are a contract with every future reader; change them deliberately.

## Directives

- Choose chunks to match the dominant access pattern (whole-row reads want row-aligned chunks) and state them explicitly at creation; aim for chunk sizes in the megabytes, not kilobytes.
- State compression explicitly at array creation rather than relying on defaults that may drift across versions.
- Organize related arrays in groups with attributes (`.attrs`) carrying the units and provenance a reader needs; attrs are JSON, so keep values plain.
- Write `zarr_format=2` explicitly when older readers (or consumers pinned to v2) must read the store; default new stores to v3.
- Consolidate metadata (`zarr.consolidate_metadata`) for read-heavy stores on object storage or high-latency filesystems; skip it for hierarchies that change often, since every update must re-consolidate (and the feature is still marked experimental for v3 stores).
- Align dask/xarray chunking with the store's chunks when reading; mismatched chunking silently multiplies I/O.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `zarr.DirectoryStore(path)` | `zarr.storage.LocalStore(path)` (or pass the path directly) |
| `zarr.open(...)` with v2-only kwargs (`synchronizer=`) | the v3 API; coordinate writers externally |
| `numcodecs.Blosc(...)` passed as `compressor=` to v3 arrays | `compressors=[zarr.codecs.BloscCodec(...)]` on `zarr.create_array` |
| assuming v2 on-disk layout (`.zarray`, `.zgroup`) in new stores | v3 layout (`zarr.json`); pass `zarr_format=2` when compatibility demands it |
