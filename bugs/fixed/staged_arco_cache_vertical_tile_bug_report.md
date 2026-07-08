# Bug Report: Staged ARCO Cache Tiles Do Not Correctly Distinguish Vertical Coverage

## Summary

The staged ARCO cache currently has ambiguous behaviour when deciding whether a local Zarr tile already satisfies a requested budget run. The immediate issue is that the staging script can skip retrieval for a new request if an existing tile overlaps the requested time and horizontal region, even when that tile was created for a different vertical control volume.

This is incorrect for Eulerian heat-budget runs because the vertical control volume is part of the scientific definition of the dataset. A tile staged for a full-column or near-surface budget is not equivalent to a tile staged for a smaller free-tropospheric layer such as 300–500 hPa.

The cache system should treat the following as part of tile identity and coverage:

- horizontal region / bounding box;
- time interval;
- top pressure boundary;
- bottom boundary mode, either `surface_pressure` or `pressure_level`;
- bottom pressure when `zg_bottom == "pressure_level"`;
- benchmark-variable inclusion flag;
- source dataset identity / ARCO path.

## Affected functionality

### Primary affected script

- `scripts/staged_arco_retrieval.py`

### Primary affected module

- `src_arco/cache.py`

### Related selection logic

- `src_arco/selection.py`

## Current behaviour

The staged retrieval script currently checks whether the cache already has coverage for a requested window before writing a tile. The relevant control flow is:

```python
if cache.cache_has_coverage(
    cache_root,
    source_cfg,
    request,
    include_benchmark_variables=args.include_benchmark_variables,
):
    print(f"[info] cache already covers {time_start} to {time_end}; skipping")
    continue
```

This uses broad coverage semantics. It asks whether any existing tile or combination of tiles can satisfy the request, rather than whether the exact requested tile already exists.

The cache database records `level_min` and `level_max`, but the candidate-tile query only filters by benchmark flag, time overlap, latitude overlap, and longitude overlap. It does not filter by vertical coverage. As a result, a tile for a different vertical layer can be treated as a valid candidate during coverage checks.

The cache loader then opens all candidate Zarr stores and combines them using xarray `combine_first()` semantics. If multiple tiles contain overlapping coordinates and variables, the first non-null value wins. There is currently no explicit conflict detection, no numerical-tolerance check, and no preference for exact vertical coverage.

## Why this is incorrect

The vertical bounds define the physical control volume. For the heat-budget calculation, a 300–500 hPa tile and a surface-to-top tile are not interchangeable, even if they overlap horizontally and temporally.

The current behaviour can lead to two related problems:

1. **False staging skip**  
   A request for a new vertical layer may be skipped because an older tile overlaps in time and horizontal space.

2. **Ambiguous loading from overlapping tiles**  
   If multiple tiles contain overlapping data, `combine_first()` silently selects one copy based on candidate ordering. This is not a scientifically explicit rule, especially if two overlapping tiles contain slightly different numerical values due to encoding, rechunking, or future preprocessing changes.

## Intended behaviour

### Tile identity

A staged Zarr tile should be unique to the full request identity, including the vertical control-volume definition.

At minimum, tile identity should include:

```text
source kind
ARCO source path
time_start
time_end
bbox
margin_n
zg_top_pressure
zg_bottom
zg_bottom_pressure
include_benchmark_variables
cache schema version
```

The existing `tile_id_for_request()` function already hashes the full `DomainRequest`, which includes `zg_top_pressure`, `zg_bottom`, and `zg_bottom_pressure`. This means the exact tile-ID mechanism is mostly correct. The bug is that staged retrieval does not use exact tile existence as its skip condition.

### Staged retrieval

During staging, the script should only skip if the exact tile already exists.

Expected behaviour:

- same source, time, region, vertical bounds, and benchmark flag: skip;
- same source, time, and region but different vertical bounds: stage a new tile;
- same source, region, and vertical bounds but different time interval: stage a new tile;
- same source, time, region, and vertical bounds but different benchmark inclusion: stage a new tile or satisfy benchmark requests only from benchmark-capable tiles.

### Cache loading

When loading from the staged cache, the loader should prefer exact or minimal covering tiles rather than relying on broad overlap and `combine_first()` ordering.

Expected loading policy:

1. Prefer an exact tile match for the requested source/time/region/vertical bounds/benchmark flag.
2. If no exact tile exists, search for tiles that fully cover the requested time, horizontal region, and vertical interval.
3. Prefer the smallest sufficient covering tile or a deterministic set of minimal tiles.
4. Only combine multiple tiles when their coverage is complementary, not redundant.
5. If overlapping tiles are combined, check overlapping values against a tolerance and raise a clear error if conflicts exceed the tolerance.

## Proposed immediate fix

### Change staged retrieval skip logic

Replace the current `cache.cache_has_coverage(...)` skip check in `scripts/staged_arco_retrieval.py` with an exact tile-path check.

Suggested helper:

```python
def _exact_tile_path(
    cache_root: Path,
    source_cfg: specs.DataSourceConfig,
    request: specs.DomainRequest,
    *,
    include_benchmark_variables: bool,
) -> Path:
    tile_id = cache.tile_id_for_request(
        source_cfg,
        request,
        include_benchmark_variables=include_benchmark_variables,
    )
    return cache_root / cache.TILES_DIR / f"{tile_id}.zarr"
```

Suggested use in the staging loop:

```python
tile_path = _exact_tile_path(
    cache_root,
    source_cfg,
    request,
    include_benchmark_variables=args.include_benchmark_variables,
)
if tile_path.exists():
    print(
        f"[info] exact staged tile already exists for "
        f"{time_start} to {time_end}; skipping"
    )
    continue
```

This prevents a broad or differently sliced tile from blocking creation of a new tile for a distinct vertical control volume.

## Proposed loading-side fix

### Add vertical coverage filtering

The SQLite index already stores `level_min` and `level_max`. Candidate tile selection should use these fields to exclude tiles that cannot cover the requested vertical interval.

For `zg_bottom == "pressure_level"`, candidate tiles should satisfy:

```text
level_min <= zg_top_pressure
level_max >= zg_bottom_pressure
```

where `level_min` is the smallest pressure level available in the tile and `level_max` is the largest pressure level available in the tile.

For `zg_bottom == "surface_pressure"`, candidate logic is more subtle because the bottom boundary follows `sp`. In that case, the cache should ensure that the tile includes all pressure levels needed from `zg_top_pressure` down to the lowest staged pressure level used for surface-following calculations. A conservative policy is to require full vertical coverage from `zg_top_pressure` to the dataset's deepest required pressure level for the staged tile.

### Prefer exact or minimal coverage

Candidate selection should not simply order by `created_at ASC`. Instead, it should prefer:

1. exact tile ID match;
2. exact vertical match;
3. smallest sufficient vertical range;
4. smallest sufficient horizontal range;
5. deterministic tie-breaker, such as newest schema version then created time.

This avoids silently using an old broad tile when a newer, more precise tile exists.

## Redundancy issue

The exact-tile fix may create duplicate physical information across multiple Zarr tiles. For example, a full-column tile and a 300–500 hPa tile for the same region/time will both contain data for 300–500 hPa.

This is not intrinsically unsafe if loading rules are deterministic and conflict-checked, but it is inefficient. Zarr does not automatically deduplicate data across independent stores. Each tile is its own `.zarr` directory, so duplicated coordinate/variable chunks are physically stored more than once.

## Proposed post-download redundancy-reduction tool

Rather than making `staged_arco_retrieval.py` responsible for complicated incremental cache maintenance, create a separate cache-maintenance tool.

Suggested script name:

```text
scripts/compact_staged_arco_cache.py
```

Possible responsibilities:

1. Inspect the SQLite cache index and all tile metadata.
2. Identify overlapping tiles in time, horizontal region, vertical levels, and variables.
3. Detect exact duplicate coordinate/variable coverage.
4. Optionally compare overlapping values with a configurable tolerance.
5. Rewrite redundant tiles into a smaller set of canonical tiles.
6. Update the SQLite index atomically after successful rewrite.
7. Optionally archive or delete superseded tile stores.

Possible command-line modes:

```text
--dry-run
    Report redundant coverage without changing files.

--check-overlaps
    Compare overlapping data values and report max absolute differences.

--compact
    Rewrite the cache into a reduced set of nonredundant tiles.

--delete-superseded
    Remove redundant old tiles after successful compaction.
```

This keeps staged retrieval simple: retrieval writes exact requested tiles. Cache compaction becomes a separate maintenance step that can be run after a production staging campaign.

## Suggested implementation phases

### Phase 1: Correct staging semantics

- Change `staged_arco_retrieval.py` to skip only exact existing tiles.
- Keep broad coverage loading unchanged temporarily.
- This fixes the immediate problem of vertical requests being skipped incorrectly.

### Phase 2: Correct candidate selection

- Add vertical coverage filters to `_candidate_tile_paths()`.
- Prefer exact or minimal covering tiles.
- Add explicit conflict detection for overlapping candidates.

### Phase 3: Add cache diagnostics

- Add a diagnostic command that lists tiles and their time/horizontal/vertical/benchmark coverage.
- Add a command that reports overlapping tile regions.
- Add a command that checks whether a request is satisfied exactly, broadly, or not at all.

### Phase 4: Add cache compaction

- Implement a post-download compaction tool.
- Keep staged retrieval logic simple and robust.
- Use compaction only when redundancy becomes operationally expensive.

## Acceptance criteria

The bug should be considered fixed when:

1. Running staged retrieval for `ocean_test`, May–September 1941, surface/full-column settings creates or uses one exact tile.
2. Running staged retrieval again for `ocean_test`, May–September 1941, `--zg-bottom pressure_level --zg-top-pa 30000 --zg-bottom-pa 50000` creates or uses a distinct exact tile.
3. The second run is not skipped merely because the first tile overlaps horizontally and temporally.
4. Loading from the staged cache prefers the exact 300–500 hPa tile for a 300–500 hPa request.
5. If two candidate tiles overlap and contain conflicting data, the loader either selects according to an explicit deterministic policy or raises a clear conflict error.
6. A cache-inspection tool can report redundant overlap between tiles.

## Notes

The immediate fix should be conservative. It is better to write a few redundant tiles than to silently skip a scientifically distinct vertical control volume or silently load values from an unintended tile.

The redundancy problem is real, but it should likely be addressed by a post-download cache-maintenance tool rather than by making the staging retrieval code responsible for incremental spatial/vertical differencing.
