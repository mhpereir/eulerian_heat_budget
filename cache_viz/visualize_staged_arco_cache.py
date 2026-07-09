#!/usr/bin/env python3
"""Create read-only visual summaries of an indexed staged ARCO cache."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src import config
except Exception:
    config = None


CACHE_DB_NAME = "cache.sqlite"

CSV_COLUMNS = [
    "tile_id",
    "path",
    "tile_exists",
    "time_start",
    "time_end",
    "region",
    "bbox",
    "lat_min",
    "lat_max",
    "lon_min",
    "lon_max",
    "level_min_pa",
    "level_max_pa",
    "level_range_hpa",
    "vertical_domain",
    "vertical_domain_label",
    "zg_top_pressure_pa",
    "zg_bottom",
    "zg_bottom_pressure_pa",
    "include_benchmark",
    "source_kind",
    "arco_path",
    "created_at",
]

VERTICAL_DOMAIN_ORDER = {
    "full_atmosphere": 0,
    "atmosphere_only": 1,
    "surface_700hpa": 2,
    "surface_pressure": 3,
    "unknown": 9,
}

VERTICAL_SHADE = {
    "full_atmosphere": 0.05,
    "atmosphere_only": 0.36,
    "surface_700hpa": 0.62,
    "surface_pressure": 0.74,
    "unknown": 0.20,
}

REGION_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass(frozen=True)
class Tile:
    tile_id: str
    path: str
    tile_exists: bool
    time_start: str | None
    time_end: str | None
    start_dt: datetime | None
    end_dt: datetime | None
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    level_min_pa: float
    level_max_pa: float
    include_benchmark: bool
    request: dict[str, Any]
    source: dict[str, Any]
    created_at: str
    region: str
    bbox: str
    vertical_domain: str
    vertical_domain_label: str
    source_kind: str
    arco_path: str

    @property
    def vertical_order(self) -> int:
        return VERTICAL_DOMAIN_ORDER.get(self.vertical_domain, VERTICAL_DOMAIN_ORDER["unknown"])

    @property
    def level_range_hpa(self) -> str:
        return f"{_pa_to_hpa(self.level_min_pa):g}-{_pa_to_hpa(self.level_max_pa):g}"


@dataclass(frozen=True)
class TimelineGroup:
    region: str
    bbox: str
    vertical_domain: str
    vertical_domain_label: str
    level_min_pa: float
    level_max_pa: float
    include_benchmark: bool
    tiles: list[Tile]

    @property
    def vertical_order(self) -> int:
        return VERTICAL_DOMAIN_ORDER.get(self.vertical_domain, VERTICAL_DOMAIN_ORDER["unknown"])

    @property
    def label(self) -> str:
        benchmark = "benchmark" if self.include_benchmark else "budget"
        tile_word = "tile" if len(self.tiles) == 1 else "tiles"
        return f"{self.region} | {self.vertical_domain_label} | {benchmark} | {len(self.tiles)} {tile_word}"


def main() -> None:
    args = _parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / cache_root.name
    output_dir = output_dir.expanduser().resolve()

    tiles = read_tiles(cache_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "tiles.csv"
    timeline_path = output_dir / "coverage_timeline.png"

    write_tiles_csv(tiles, csv_path)
    write_coverage_timeline(
        tiles,
        timeline_path,
        title=args.title or f"Staged ARCO cache coverage: {cache_root.name}",
    )

    print(f"wrote {csv_path}")
    print(f"wrote {timeline_path}")
    print(f"indexed tiles: {len(tiles)}")
    missing = sum(1 for tile in tiles if not tile.tile_exists)
    if missing:
        print(f"indexed tiles with missing zarr paths: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read cache.sqlite from a staged ARCO cache and write tiles.csv plus "
            "coverage_timeline.png. The cache database is opened read-only."
        )
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Indexed staged ARCO cache root containing cache.sqlite and tiles/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for tiles.csv and coverage_timeline.png. "
            "Default: cache_viz/outputs/<cache-root-name>/."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for coverage_timeline.png.",
    )
    return parser.parse_args()


def read_tiles(cache_root: Path) -> list[Tile]:
    db_path = cache_root / CACHE_DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(f"staged ARCO cache index not found: {db_path}")

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT tile_id, path, time_start, time_end, lat_min, lat_max,
                   lon_min, lon_max, level_min, level_max, include_benchmark,
                   request_json, source_json, created_at
            FROM tiles
            ORDER BY lat_min, lon_min, level_min, time_start, tile_id
            """
        ).fetchall()

    return [_tile_from_row(cache_root, row) for row in rows]


def write_tiles_csv(tiles: list[Tile], path: Path) -> None:
    sorted_tiles = sorted(tiles, key=_tile_sort_key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for tile in sorted_tiles:
            writer.writerow(_tile_csv_row(tile))


def write_coverage_timeline(tiles: list[Tile], path: Path, *, title: str) -> None:
    groups = _timeline_groups(tiles)
    if not groups:
        raise ValueError("No tiles were found in the staged ARCO cache index.")

    region_colors = _region_color_map([group.region for group in groups])
    y_positions: list[float] = []
    y_ticks: list[float] = []
    y_labels: list[str] = []

    y = 0.0
    last_region: str | None = None
    region_boundaries: list[float] = []
    for group in groups:
        if last_region is not None and group.region != last_region:
            region_boundaries.append(y - 0.5)
            y += 0.55
        y_positions.append(y)
        y_ticks.append(y)
        y_labels.append(group.label)
        y += 1.0
        last_region = group.region

    fig_height = max(3.6, 1.2 + 0.42 * len(groups))
    fig_width = 14.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    min_width_days = 1.0 / 24.0
    for group, y_pos in zip(groups, y_positions):
        base_color = region_colors[group.region]
        facecolor = _shade_color(base_color, VERTICAL_SHADE.get(group.vertical_domain, 0.2))
        for tile in sorted(group.tiles, key=_tile_sort_key):
            if tile.start_dt is None or tile.end_dt is None:
                continue
            start_num = mdates.date2num(tile.start_dt)
            end_num = mdates.date2num(tile.end_dt)
            width = max(end_num - start_num, min_width_days)
            edgecolor = "#222222" if not tile.tile_exists else "white"
            hatch = "///" if not tile.tile_exists else None
            ax.broken_barh(
                [(start_num, width)],
                (y_pos - 0.34, 0.68),
                facecolors=facecolor,
                edgecolors=edgecolor,
                linewidth=0.7,
                hatch=hatch,
            )

    for boundary in region_boundaries:
        ax.axhline(boundary, color="#c8c8c8", linewidth=0.8)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Tile time coverage")
    ax.set_title(title, loc="left", fontsize=12)

    region_legend = [
        Patch(facecolor=color, edgecolor="none", label=region)
        for region, color in region_colors.items()
    ]
    vertical_legend = [
        Patch(
            facecolor=_shade_color("#606060", VERTICAL_SHADE[key]),
            edgecolor="none",
            label=label,
        )
        for key, label in [
            ("full_atmosphere", "full atmosphere"),
            ("atmosphere_only", "atmosphere only"),
            ("surface_700hpa", "surface-700 hPa"),
        ]
    ]
    missing_legend = [Patch(facecolor="white", edgecolor="#222222", hatch="///", label="missing zarr path")]
    handles = region_legend + vertical_legend + missing_legend
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=8,
    )

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _tile_from_row(cache_root: Path, row: sqlite3.Row) -> Tile:
    request = _loads_json(row["request_json"])
    source = _loads_json(row["source_json"])
    lat_min = float(row["lat_min"])
    lat_max = float(row["lat_max"])
    lon_min = float(row["lon_min"])
    lon_max = float(row["lon_max"])
    level_min = float(row["level_min"])
    level_max = float(row["level_max"])
    bbox = _bbox_label(lat_min, lat_max, lon_min, lon_max)
    region = _region_name(lat_min, lat_max, lon_min, lon_max)
    vertical_domain, vertical_label = _vertical_domain(row, request)
    rel_path = str(row["path"])
    tile_path = cache_root / rel_path

    return Tile(
        tile_id=str(row["tile_id"]),
        path=rel_path,
        tile_exists=tile_path.exists(),
        time_start=row["time_start"],
        time_end=row["time_end"],
        start_dt=_parse_datetime(row["time_start"]),
        end_dt=_parse_datetime(row["time_end"]),
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        level_min_pa=level_min,
        level_max_pa=level_max,
        include_benchmark=bool(row["include_benchmark"]),
        request=request,
        source=source,
        created_at=str(row["created_at"]),
        region=region,
        bbox=bbox,
        vertical_domain=vertical_domain,
        vertical_domain_label=vertical_label,
        source_kind=str(source.get("kind", "")),
        arco_path=str(source.get("arco_path", "")),
    )


def _tile_csv_row(tile: Tile) -> dict[str, Any]:
    return {
        "tile_id": tile.tile_id,
        "path": tile.path,
        "tile_exists": int(tile.tile_exists),
        "time_start": tile.time_start or "",
        "time_end": tile.time_end or "",
        "region": tile.region,
        "bbox": tile.bbox,
        "lat_min": tile.lat_min,
        "lat_max": tile.lat_max,
        "lon_min": tile.lon_min,
        "lon_max": tile.lon_max,
        "level_min_pa": tile.level_min_pa,
        "level_max_pa": tile.level_max_pa,
        "level_range_hpa": tile.level_range_hpa,
        "vertical_domain": tile.vertical_domain,
        "vertical_domain_label": tile.vertical_domain_label,
        "zg_top_pressure_pa": _optional_number(tile.request.get("zg_top_pressure")),
        "zg_bottom": tile.request.get("zg_bottom", ""),
        "zg_bottom_pressure_pa": _optional_number(tile.request.get("zg_bottom_pressure")),
        "include_benchmark": int(tile.include_benchmark),
        "source_kind": tile.source_kind,
        "arco_path": tile.arco_path,
        "created_at": tile.created_at,
    }


def _timeline_groups(tiles: list[Tile]) -> list[TimelineGroup]:
    grouped: dict[tuple[Any, ...], list[Tile]] = {}
    for tile in tiles:
        key = (
            tile.region,
            tile.bbox,
            tile.vertical_order,
            tile.vertical_domain,
            tile.vertical_domain_label,
            tile.level_min_pa,
            tile.level_max_pa,
            tile.include_benchmark,
        )
        grouped.setdefault(key, []).append(tile)

    groups = [
        TimelineGroup(
            region=key[0],
            bbox=key[1],
            vertical_domain=key[3],
            vertical_domain_label=key[4],
            level_min_pa=key[5],
            level_max_pa=key[6],
            include_benchmark=key[7],
            tiles=group_tiles,
        )
        for key, group_tiles in grouped.items()
    ]
    return sorted(groups, key=_group_sort_key)


def _group_sort_key(group: TimelineGroup) -> tuple[Any, ...]:
    return (
        _region_sort_value(group.region),
        group.region,
        group.bbox,
        group.vertical_order,
        group.level_min_pa,
        group.level_max_pa,
        int(group.include_benchmark),
    )


def _tile_sort_key(tile: Tile) -> tuple[Any, ...]:
    return (
        _region_sort_value(tile.region),
        tile.region,
        tile.bbox,
        tile.vertical_order,
        tile.level_min_pa,
        tile.level_max_pa,
        int(tile.include_benchmark),
        tile.start_dt or datetime.min,
        tile.end_dt or datetime.min,
        tile.tile_id,
    )


def _region_sort_value(region: str) -> int:
    if config is None:
        return 10_000
    region_names = list(config.REGIONS)
    try:
        return region_names.index(region)
    except ValueError:
        return 10_000


def _region_color_map(regions: list[str]) -> dict[str, str]:
    unique_regions = sorted(set(regions), key=lambda region: (_region_sort_value(region), region))
    return {
        region: REGION_PALETTE[index % len(REGION_PALETTE)]
        for index, region in enumerate(unique_regions)
    }


def _region_name(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    if config is not None:
        for name, bbox in config.REGIONS.items():
            if _bbox_matches((lat_min, lat_max, lon_min, lon_max), bbox):
                return name
    return f"bbox {lat_min:g}:{lat_max:g}, {lon_min:g}:{lon_max:g}"


def _bbox_matches(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(left, right))


def _bbox_label(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    return f"{lat_min:g}:{lat_max:g},{lon_min:g}:{lon_max:g}"


def _vertical_domain(row: sqlite3.Row, request: dict[str, Any]) -> tuple[str, str]:
    zg_bottom = request.get("zg_bottom")
    zg_top = _optional_float(request.get("zg_top_pressure"))
    zg_bottom_pressure = _optional_float(request.get("zg_bottom_pressure"))
    level_min = float(row["level_min"])
    level_max = float(row["level_max"])

    if zg_bottom == "surface_pressure":
        top_pa = zg_top if zg_top is not None else level_min
        if top_pa <= 1_000.0 and level_max >= 90_000.0:
            return "full_atmosphere", "full atmosphere"
        if abs(top_pa - 70_000.0) <= 1.0:
            return "surface_700hpa", "surface-700 hPa"
        return "surface_pressure", f"surface-{_pa_to_hpa(top_pa):g} hPa"

    if zg_bottom == "pressure_level":
        top_pa = zg_top if zg_top is not None else level_min
        bottom_pa = zg_bottom_pressure if zg_bottom_pressure is not None else level_max
        top_hpa = _pa_to_hpa(top_pa)
        bottom_hpa = _pa_to_hpa(bottom_pa)
        if abs(top_hpa - bottom_hpa) <= 1e-9:
            return "atmosphere_only", f"atmosphere only ({top_hpa:g} hPa)"
        return "atmosphere_only", f"atmosphere only ({top_hpa:g}-{bottom_hpa:g} hPa)"

    return "unknown", f"{_pa_to_hpa(level_min):g}-{_pa_to_hpa(level_max):g} hPa"


def _shade_color(base_color: str, white_mix: float) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(base_color)
    white_mix = max(0.0, min(1.0, white_mix))
    return tuple((1.0 - white_mix) * channel + white_mix for channel in rgb)


def _loads_json(value: str) -> dict[str, Any]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object, got {type(loaded).__name__}")
    return loaded


def _parse_datetime(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        prefix, suffix = text.split(".", 1)
        tz = ""
        for marker in ("+", "-"):
            if marker in suffix:
                frac, tz_tail = suffix.split(marker, 1)
                suffix = frac
                tz = marker + tz_tail
                break
        text = f"{prefix}.{suffix[:6].ljust(6, '0')}{tz}"
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _optional_number(value: Any) -> float | str:
    parsed = _optional_float(value)
    return "" if parsed is None else parsed


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pa_to_hpa(value: float) -> float:
    return float(value) / 100.0


if __name__ == "__main__":
    main()
