'''
Docstring for eulerian_heat_budget.src.grid

Computes grid metrics: # Layer A

- `cell_area(lat, lon)`
- Optional `dx`, `dy`
- Handles spherical Earth geometry

Must be deterministic and independently testable.
'''

import xarray as xr
import numpy as np
from typing import Tuple

from . import config


def _cell_edges_from_centers(coord: xr.DataArray, name: str) -> np.ndarray:
    """Build 1D cell edges from 1D center coordinates.

    Accepts either strictly increasing or strictly decreasing coordinates.
    """
    values = np.asarray(coord.values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} coordinate must be one-dimensional.")
    if values.size < 2:
        raise ValueError(f"{name} coordinate must contain at least two points.")

    diffs = np.diff(values)
    if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
        raise ValueError(f"{name} coordinate must be strictly monotonic.")

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * diffs[0]
    edges[-1] = values[-1] + 0.5 * diffs[-1]
    return edges


def _interval_bounds_from_full_cell_starts(coord: xr.DataArray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return interval start/end arrays when coordinates represent full-cell starts.

    For input start coordinates s[0..N-1], intervals are [s[i], s[i+1]] for i=0..N-2.
    """
    values = np.asarray(coord.values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} coordinate must be one-dimensional.")
    if values.size < 2:
        raise ValueError(f"{name} coordinate must contain at least two points.")

    diffs = np.diff(values)
    if not np.all(diffs > 0.0):
        raise ValueError(f"{name} coordinate must be strictly increasing.")

    return values[:-1], values[1:]


def _interval_bounds_from_centers(coord: xr.DataArray, name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return interval start/end arrays when coordinates represent cell centers."""
    edges = _cell_edges_from_centers(coord, name)
    return edges[:-1], edges[1:]


def determine_domain(
    ds: xr.Dataset,
    *,
    level_name: str = "level",
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.Dataset:
    """
    Return a domain-cropped dataset with coordinate metadata for cell bounds.

    Horizontal coordinates are interpreted as cell-start coordinates (length N_start).
    We build cells from consecutive starts (N_cells = N_start - 1), then trim `margin`
    cells from each edge. After this, ds_domain[lat]/[lon] are cell centers (Option A),
    and bounds are in lat_start/lat_end, lon_start/lon_end.
    """
    margin = int(config.margin)
    if margin < 0:
        raise ValueError("config.margin must be non-negative.")

    for name in (level_name, lat_name, lon_name):
        if name not in ds.coords:
            raise ValueError(f"Dataset must contain '{name}' coordinate.")

    lat = ds[lat_name]
    lon = ds[lon_name]
    level = ds[level_name]

    Nlat_start = lat.size
    Nlon_start = lon.size

    # Number of *cells* implied by starts
    Nlat_cell = Nlat_start - 1
    Nlon_cell = Nlon_start - 1

    # Margin is in cells
    if (Nlat_cell - 2 * margin) <= 0 or (Nlon_cell - 2 * margin) <= 0:
        raise ValueError("config.margin is too large for current grid cell dimensions.")

    # Slice *cells* (i indexes the start of each cell)
    lat_cell_slice = slice(margin, Nlat_cell - margin)
    lon_cell_slice = slice(margin, Nlon_cell - margin)

    # Subset dataset on the same slices (dims remain lat/lon, but now represent cells)
    ds_domain = ds.isel({lat_name: lat_cell_slice, lon_name: lon_cell_slice}).copy()

    # Build bounds without extrapolation: start[i] and start[i+1]
    lat_start = lat.isel({lat_name: lat_cell_slice}).values
    lat_end   = lat.isel({lat_name: slice(margin + 1, margin + 1 + lat_start.size)}).values
    lat_mid   = 0.5 * (lat_start + lat_end)

    lon_start = lon.isel({lon_name: lon_cell_slice}).values
    lon_end   = lon.isel({lon_name: slice(margin + 1, margin + 1 + lon_start.size)}).values
    lon_mid   = 0.5 * (lon_start + lon_end)

    # Vertical: level is already midpoints; make edges for bounds
    p_edges = _cell_edges_from_centers(level, level_name)
    p_mid   = np.asarray(level.values, dtype=float)

    # Traceability indices as coords on the *same* dims
    lat_cell_id = np.arange(margin, margin + lat_mid.size, dtype=int)
    lon_cell_id = np.arange(margin, margin + lon_mid.size, dtype=int)
    p_cell_id   = np.arange(p_mid.size, dtype=int)

    lat_start = lat_start.astype(np.float64)
    lat_end   = lat_end.astype(np.float64)
    lat_mid   = lat_mid.astype(np.float64)
    lon_start = lon_start.astype(np.float64)
    lon_end   = lon_end.astype(np.float64)
    lon_mid   = lon_mid.astype(np.float64)
    p_edges   = p_edges.astype(np.float64)
    p_mid     = p_mid.astype(np.float64)

    # Option A: overwrite lat/lon coords to be centers; attach bounds on same dims
    ds_domain = ds_domain.assign_coords(
        {
            lat_name: (lat_name, lat_mid),
            lon_name: (lon_name, lon_mid),

            "lat_cell_id": (lat_name, lat_cell_id),
            "lon_cell_id": (lon_name, lon_cell_id),

            "lat_start": (lat_name, lat_start),
            "lat_end":   (lat_name, lat_end),
            "lon_start": (lon_name, lon_start),
            "lon_end":   (lon_name, lon_end),

            "p_cell_id": (level_name, p_cell_id),
            "p_start":   (level_name, p_edges[:-1]),
            "p_end":     (level_name, p_edges[1:]),
            "p_mid":     (level_name, p_mid),
        }
    )

    ds_domain.attrs.update(
        {
            "lat_min": float(lat_start[0]),
            "lat_max": float(lat_end[-1]),
            "lon_min": float(lon_start[0]),
            "lon_max": float(lon_end[-1]),
            "margin": margin,
            "horizontal_coord_type": "cell_center_with_bounds",
            "horizontal_input_interpretation": "cell_starts",
            "vertical_coord_type": "level_center_with_bounds",
            "vertical_input_interpretation": "level_centers",
        }
    )

    return ds_domain



def get_horizontal_cell_areas(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute geometric horizontal cell areas for the (lat, lon) cell grid.

    Assumptions / Contract
    ----------------------
    - ds has been processed by determine_domain():
      - dims: level, lat, lon are *cell* dimensions
      - ds[lat], ds[lon] are *cell centers*
      - bounds exist as coords on same dims:
        lat_start(lat), lat_end(lat), lon_start(lon), lon_end(lon)

    Returns
    -------
    xr.DataArray with variable 
        - top(lat, lon); units of m^2.
    """
    # ---- Required coordinates ----
    for k in ("lat_start", "lat_end", "lon_start", "lon_end"):
        if k not in ds.coords:
            raise ValueError(f"Dataset must contain coordinate '{k}' from determine_domain().")

    lat_name = "lat"
    lon_name = "lon"
    if lat_name not in ds.dims or lon_name not in ds.dims:
        raise ValueError("Expected ds to have dims ('lat','lon') as cell dimensions.")

    # ---- Geometry ----
    lat_start_rad = np.deg2rad(ds["lat_start"])
    lat_end_rad   = np.deg2rad(ds["lat_end"])
    lon_start_rad = np.deg2rad(ds["lon_start"])
    lon_end_rad   = np.deg2rad(ds["lon_end"])

    d_sin_lat = np.abs(np.sin(lat_end_rad) - np.sin(lat_start_rad))  # (lat,)
    d_lon     = np.abs(lon_end_rad - lon_start_rad)                  # (lon,)

    cell_area = (config.R_earth ** 2) * d_sin_lat * d_lon             # broadcast -> (lat, lon)
    cell_area = cell_area.rename("top") # type: ignore

    cell_area.attrs.update(
        {
            "units": "m2",
            "long_name": "Top horizontal wall geometric area",
            "horizontal_coord_type": ds.attrs.get("horizontal_coord_type"),
            "horizontal_input_interpretation": ds.attrs.get("horizontal_input_interpretation"),
            "longitude_closure": "open_domain",
        }
    )

    # Ensure key coords are attached (xarray usually keeps them, but be explicit)
    cell_area = cell_area.assign_coords(
        {
            "lat_start": ds["lat_start"],
            "lat_end":   ds["lat_end"],
            "lon_start": ds["lon_start"],
            "lon_end":   ds["lon_end"],
            **({"lat_cell_id": ds["lat_cell_id"]} if "lat_cell_id" in ds.coords else {}),
            **({"lon_cell_id": ds["lon_cell_id"]} if "lon_cell_id" in ds.coords else {}),
        }
    )

    return cell_area

def get_vertical_cell_areas(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute geometric vertical wall areas for a lat/lon bounding-box control volume.

    Assumptions / Contract
    ----------------------
    - ds has been processed by determine_domain():
      - dims: level, lat, lon are *cell* dimensions
      - ds[lat], ds[lon] are *cell centers*
      - bounds exist as coords on same dims:
        lat_start(lat), lat_end(lat), lon_start(lon), lon_end(lon)
      - vertical bounds exist as coords on level dim:
        p_start(level), p_end(level)

    Returns
    -------
    xr.Dataset with variables (units m*Pa):
      - east(level, lat)  : wall at lon = lon_max
      - west(level, lat)  : wall at lon = lon_min
      - south(level, lon) : wall at lat = lat_min
      - north(level, lon) : wall at lat = lat_max
    """
    # Required coordinates
    for k in ("p_start", "p_end", "lat_start", "lat_end", "lon_start", "lon_end"):
        if k not in ds.coords:
            raise ValueError(f"Dataset must contain coordinate '{k}' from determine_domain().")

    # Domain bounds (already cropped)
    lat_min = ds.attrs.get("lat_min")
    lat_max = ds.attrs.get("lat_max")
    lon_min = ds.attrs.get("lon_min")
    lon_max = ds.attrs.get("lon_max")
    if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
        raise ValueError("Dataset must contain 'lat_min', 'lat_max', 'lon_min', 'lon_max' attributes.")

    # Vertical thickness (Pa): (level,)
    dp = np.abs(ds["p_end"] - ds["p_start"])

    # Horizontal cell widths
    # dy(lat) = R * dphi
    dphi = np.abs(np.deg2rad(ds["lat_end"]) - np.deg2rad(ds["lat_start"]))  # (lat,)
    dy = config.R_earth * dphi  # (lat,)

    # dx(lon) along constant-lat boundary, using each lon cell's dlon
    dlon = np.abs(np.deg2rad(ds["lon_end"]) - np.deg2rad(ds["lon_start"]))  # (lon,)

    phi_south = np.deg2rad(float(lat_min))
    phi_north = np.deg2rad(float(lat_max))
    dx_south = config.R_earth * np.cos(phi_south) * dlon  # (lon,)
    dx_north = config.R_earth * np.cos(phi_north) * dlon  # (lon,)

    # Broadcast to 2D walls (level, lat) and (level, lon)
    east  = (dp * dy).rename("east") #type: ignore
    west  = (dp * dy).rename("west") #type: ignore
    south = (dp * dx_south).rename("south")
    north = (dp * dx_north).rename("north")

    east.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_max"]),
        "normal_convention": "geometric_only"   
    })

    west.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_min"]),
        "normal_convention": "geometric_only"
    })

    south.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_min"]),
        "normal_convention": "geometric_only"
    })

    north.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_max"]),
        "normal_convention": "geometric_only"
    })

    units = "m*Pa"
    for da, ln in [
        (east,  "East vertical wall geometric area (lon = lon_max)"),
        (west,  "West vertical wall geometric area (lon = lon_min)"),
        (south, "South vertical wall geometric area (lat = lat_min)"),
        (north, "North vertical wall geometric area (lat = lat_max)"),
    ]:
        da.attrs.update({"units": units, "long_name": ln, 
                         "horizontal_convention": ds.attrs.get("horizontal_coord_type"),
                         "horizontal_input_interpretation": ds.attrs.get("horizontal_input_interpretation"),
                         "vertical_convention": ds.attrs.get("vertical_coord_type"),
                         "vertical_input_interpretation": ds.attrs.get("vertical_input_interpretation"),
    })

    out = xr.Dataset({"east": east, "west": west, "south": south, "north": north})

    # Carry through coordinate metadata so consumers can locate bounds easily
    # (They’re already in ds.coords; xarray will propagate coords for used dims.)
    out = out.assign_coords(
        {
            "p_start": ds["p_start"],
            "p_end": ds["p_end"],
            "p_mid": ds.get("p_mid", ds["level"]),
            "lat_start": ds["lat_start"],
            "lat_end": ds["lat_end"],
            "lon_start": ds["lon_start"],
            "lon_end": ds["lon_end"],
            # Optional traceability ids if present
            **({ "lat_cell_id": ds["lat_cell_id"] } if "lat_cell_id" in ds.coords else {}),
            **({ "lon_cell_id": ds["lon_cell_id"] } if "lon_cell_id" in ds.coords else {}),
        }
    )

    out.attrs.update(
        {
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "area_units": units,
            "horizontal_coord_type": ds.attrs.get("horizontal_coord_type"),
            "horizontal_input_interpretation": ds.attrs.get("horizontal_input_interpretation"),
            "longitude_closure": "open_domain",
        }
    )
    return out


def get_cell_volumes(ds: xr.Dataset) -> xr.DataArray:
    """Compute grid-cell volumes in geometric units of m^2*Pa (no /g normalization).

    Requires ds from determine_domain() (Option A):
      - dims: level, lat, lon are cell dims
      - bounds: lat_start/lat_end on lat dim; lon_start/lon_end on lon dim
      - vertical bounds: p_start/p_end on level dim
    """
    # ---- Required coordinates ----
    for k in ("lat_start", "lat_end", "lon_start", "lon_end", "p_start", "p_end"):
        if k not in ds.coords:
            raise ValueError(f"Dataset must contain coordinate '{k}' from determine_domain().")

    level_name = "level"
    lat_name = "lat"
    lon_name = "lon"
    if level_name not in ds.dims or lat_name not in ds.dims or lon_name not in ds.dims:
        raise ValueError("Expected ds to have dims ('level','lat','lon') as cell dimensions.")

    # ---- Vertical thickness (Pa): (level,) ----
    dp = np.abs(ds["p_end"] - ds["p_start"])
    if (dp <= 0).any():
        raise ValueError("Pressure layer thickness must be strictly positive.")

    # ---- Horizontal cell area on sphere: (lat, lon) ----
    lat_start_rad = np.deg2rad(ds["lat_start"])
    lat_end_rad   = np.deg2rad(ds["lat_end"])
    lon_start_rad = np.deg2rad(ds["lon_start"])
    lon_end_rad   = np.deg2rad(ds["lon_end"])

    d_sin_lat = np.abs(np.sin(lat_end_rad) - np.sin(lat_start_rad))   # (lat,)
    d_lon     = np.abs(lon_end_rad - lon_start_rad)                   # (lon,)

    cell_area = (config.R_earth ** 2) * d_sin_lat * d_lon              # broadcast -> (lat, lon)
    cell_area = cell_area.rename("cell_area") #type: ignore
    cell_area.attrs.update({"units": "m2", "long_name": "Horizontal grid-cell area on a spherical Earth"})

    # ---- Volume in pressure coords: A * dp -> (level, lat, lon) ----
    cell_volume = (cell_area * dp).rename("cell_volume")
    cell_volume.attrs.update(
        {
            "units": "m2*Pa",
            "long_name": "Grid-cell geometric volume in pressure coordinates (A*dp)",
            "horizontal_coord_type": ds.attrs.get("horizontal_coord_type", "cell_center_with_bounds"),
            "vertical_coord_type": ds.attrs.get("vertical_coord_type", "level_center_with_bounds"),
            "longitude_closure": "open_domain",
        }
    )

    # ---- Carry useful coords through (optional but helpful) ----
    # xarray will keep coords for the dims involved, but we can ensure key metadata is present.
    cell_volume = cell_volume.assign_coords(
        {
            "dp": dp,
            "p_start": ds["p_start"],
            "p_end": ds["p_end"],
            "lat_start": ds["lat_start"],
            "lat_end": ds["lat_end"],
            "lon_start": ds["lon_start"],
            "lon_end": ds["lon_end"],
            **({"lat_cell_id": ds["lat_cell_id"]} if "lat_cell_id" in ds.coords else {}),
            **({"lon_cell_id": ds["lon_cell_id"]} if "lon_cell_id" in ds.coords else {}),
        }
    )

    return cell_volume
