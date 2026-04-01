'''
Docstring for eulerian_heat_budget.src.grid

Geometry + domain resolution.

Key responsibilities:

- determine_domain(ds, req):
  - crops the dataset to whole grid cells consistent with the requested bounds
  - returns (ds_domain, DomainSpec)
  - constructs and attaches cell-boundary coordinates:
    - lat_start, lat_end, lon_start, lon_end
    - p_start, p_end
  - constructs and attaches cell IDs for bookkeeping (e.g., lat_cell_id, lon_cell_id, p_cell_id)
  - rewrites lat/lon to cell centers after cropping (internal convention)

This module should remain deterministic and independently testable.
'''

import xarray as xr
import numpy as np
from typing import Tuple

from . import config
from .specs import DomainRequest, DomainSpec, SurfaceBehaviour

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
    request: DomainRequest,
    *,
    level_name: str = "level",
    lat_name: str = "lat",
    lon_name: str = "lon",
    eager_loading: bool = False,
) -> Tuple[xr.Dataset, xr.Dataset, DomainSpec]:
    """
    Determine ds_domain and ds_halo assuming input lat/lon are already cell centers.

    Returns
    -------
    ds_domain : cropped interior domain
    ds_halo   : same domain with a 1-cell horizontal halo
    spec      : DomainSpec describing the resolved domain bounds
    """
    margin = int(request.margin_n)
    if margin < 1:
        raise ValueError("margin_n must be >= 1 (required to build ds_halo).")

    for name in (level_name, lat_name, lon_name):
        if name not in ds.coords:
            raise ValueError(f"Dataset must contain '{name}' coordinate.")

    lat = ds[lat_name]
    lon = ds[lon_name]
    level = ds[level_name]

    lat_vals = np.asarray(lat.values, dtype=float)
    lon_vals = np.asarray(lon.values, dtype=float)

    if not np.all(np.diff(lat_vals) > 0.0):
        raise ValueError("Latitude coordinate must be strictly increasing.")
    if not np.all(np.diff(lon_vals) > 0.0):
        raise ValueError("Longitude coordinate must be strictly increasing.")

    lat_edges = _cell_edges_from_centers(lat, lat_name)
    lon_edges = _cell_edges_from_centers(lon, lon_name)

    lat0, lat1, lon0, lon1 = map(float, request.bbox)

    # Require bbox to lie inside the coordinate coverage
    if lat0 < lat_edges[0] or lat1 > lat_edges[-1]:
        raise ValueError("Requested lat bbox is outside dataset coverage.")
    if lon0 < lon_edges[0] or lon1 > lon_edges[-1]:
        raise ValueError("Requested lon bbox is outside dataset coverage.")

    # Pressure bounds sanity checks
    if request.zg_bottom == "pressure_level":
        if request.zg_bottom_pressure is None:
            raise ValueError("zg_bottom_pressure must be specified when zg_bottom='pressure_level'.")

        p_min = float(level.min().values)
        p_max = float(level.max().values)

        if request.zg_top_pressure < p_min or request.zg_top_pressure > p_max:
            raise ValueError("Requested zg_top_pressure is outside dataset level range.")
        if request.zg_bottom_pressure < p_min or request.zg_bottom_pressure > p_max:
            raise ValueError("Requested zg_bottom_pressure is outside dataset level range.")
        if request.zg_top_pressure >= request.zg_bottom_pressure:
            raise ValueError("Requested zg_top_pressure must be less than zg_bottom_pressure.")

    # Find cells whose centers lie inside the requested bbox
    lat_mask = (lat_vals >= lat0) & (lat_vals <= lat1)
    lon_mask = (lon_vals >= lon0) & (lon_vals <= lon1)

    if not np.any(lat_mask) or not np.any(lon_mask):
        raise ValueError("No grid-cell centers fall inside requested bbox.")

    i0 = int(np.flatnonzero(lat_mask)[0])
    i1 = int(np.flatnonzero(lat_mask)[-1]) + 1  # exclusive
    j0 = int(np.flatnonzero(lon_mask)[0])
    j1 = int(np.flatnonzero(lon_mask)[-1]) + 1  # exclusive

    def _build_domain_for_margin(m: int) -> xr.Dataset:
        if m < 0:
            raise ValueError("Internal error: margin must be non-negative.")

        i0m = i0 + m
        i1m = i1 - m
        j0m = j0 + m
        j1m = j1 - m

        if i1m <= i0m or j1m <= j0m:
            raise ValueError("Domain too small after applying bbox + margin.")

        out = ds.isel(
            {
                lat_name: slice(i0m, i1m),
                lon_name: slice(j0m, j1m),
            }
        ).copy()

        lat_mid = lat_vals[i0m:i1m]
        lon_mid = lon_vals[j0m:j1m]

        lat_edges_sub = _cell_edges_from_centers(
            xr.DataArray(lat_mid, dims=(lat_name,)),
            lat_name,
        )
        lon_edges_sub = _cell_edges_from_centers(
            xr.DataArray(lon_mid, dims=(lon_name,)),
            lon_name,
        )

        # Vertical bounds from level centers
        p_edges = _cell_edges_from_centers(level, level_name)
        p_mid = np.asarray(level.values, dtype=float)

        lat_cell_id = np.arange(i0m, i1m, dtype=int)
        lon_cell_id = np.arange(j0m, j1m, dtype=int)
        p_cell_id = np.arange(p_mid.size, dtype=int)

        out = out.assign_coords(
            {
                lat_name: (lat_name, lat_mid.astype(np.float64)),
                lon_name: (lon_name, lon_mid.astype(np.float64)),
                "lat_cell_id": (lat_name, lat_cell_id),
                "lon_cell_id": (lon_name, lon_cell_id),
                "lat_start": (lat_name, lat_edges_sub[:-1].astype(np.float64)),
                "lat_end": (lat_name, lat_edges_sub[1:].astype(np.float64)),
                "lon_start": (lon_name, lon_edges_sub[:-1].astype(np.float64)),
                "lon_end": (lon_name, lon_edges_sub[1:].astype(np.float64)),
                "p_cell_id": (level_name, p_cell_id),
                "p_start": (level_name, p_edges[:-1].astype(np.float64)),
                "p_end": (level_name, p_edges[1:].astype(np.float64)),
                "p_mid": (level_name, p_mid.astype(np.float64)),
            }
        )

        out.attrs.update(
            {
                "lat_min": float(lat_edges_sub[0]),
                "lat_max": float(lat_edges_sub[-1]),
                "lon_min": float(lon_edges_sub[0]),
                "lon_max": float(lon_edges_sub[-1]),
                "margin": int(m),
                "horizontal_coord_type": "cell_center_with_bounds",
                "horizontal_input_interpretation": "cell_centers",
            }
        )

        return out

    ds_halo = _build_domain_for_margin(margin - 1)
    ds_domain = _build_domain_for_margin(margin)

    if eager_loading:
        ds_domain = ds_domain.assign(
            T=ds_domain["T"].persist(),
            w=ds_domain["w"].persist(),
            sp=ds_domain["sp"].persist(),
        )
        ds_halo = ds_halo.assign(
            sp=ds_halo["sp"].persist(),
        )

    spec = DomainSpec(
        lat_min=float(ds_domain.attrs["lat_min"]),
        lat_max=float(ds_domain.attrs["lat_max"]),
        lon_min=float(ds_domain.attrs["lon_min"]),
        lon_max=float(ds_domain.attrs["lon_max"]),
        zg_top_pressure=request.zg_top_pressure,
        zg_bottom=request.zg_bottom,
        zg_bottom_pressure=request.zg_bottom_pressure if request.zg_bottom == "pressure_level" else None,
    )

    return ds_domain, ds_halo, spec



'''
def determine_domain(
    ds: xr.Dataset,
    request: DomainRequest,
    *,
    level_name: str = "level",
    lat_name: str = "lat",
    lon_name: str = "lon",
    eager_loading: bool = False,
) -> Tuple[xr.Dataset, xr.Dataset, DomainSpec]:
    """
    Return a domain-cropped dataset with coordinate metadata for cell bounds.

    Horizontal coordinates are interpreted as cell-start coordinates (length N_start).
    We build cells from consecutive starts (N_cells = N_start - 1), then trim `margin`
    cells from each edge. After this, ds_domain[lat]/[lon] are cell centers (Option A),
    and bounds are in lat_start/lat_end, lon_start/lon_end.

    Returns:
        ds_domain : trimmed integration domain
        ds_halo   : ds_domain +1 cell pad (lat/lon)
        spec      : DomainSpec for ds_domain

    """
    margin = int(request.margin_n)
    if margin < 1:
        raise ValueError(
            "margin_n must be >= 1 (required to construct ds_halo with +1 pad)."
        )

    for name in (level_name, lat_name, lon_name):
        if name not in ds.coords:
            raise ValueError(f"Dataset must contain '{name}' coordinate.")

    lat   = ds[lat_name]
    lon   = ds[lon_name]
    level = ds[level_name]

    Nlat_cell = lat.size -1
    Nlon_cell = lon.size -1

    # Margin is in cells
    if (Nlat_cell - 2 * margin) <= 0 or (Nlon_cell - 2 * margin) <= 0:
        raise ValueError("Margin request is too large for current grid cell dimensions.")


    def _build_domain_for_margin(m: int) -> xr.Dataset:
        """Build a cell-centered dataset for a given cell-margin m, cropped to request.bbox first."""
        if m < 0:
            raise ValueError("Internal error: margin must be non-negative.")

        lat_starts = np.asarray(lat.values, dtype=float)
        lon_starts = np.asarray(lon.values, dtype=float)

        if not (np.all(np.diff(lat_starts) > 0) and np.all(np.diff(lon_starts) > 0)):
            raise ValueError("This implementation assumes lat/lon start coords are strictly increasing.")

        lat0, lat1, lon0, lon1 = map(float, request.bbox)

        # Ensure bbox lies within dataset coverage of *cells* (needs at least one cell)
        if lat0 < lat_starts[0] or lat1 > lat_starts[-1]:
            raise ValueError("Requested lat bbox is outside dataset lat start range.")
        if lon0 < lon_starts[0] or lon1 > lon_starts[-1]:
            raise ValueError("Requested lon bbox is outside dataset lon start range.")

        # Ensure pressure bounds are consistent with dataset levels
        if request.zg_bottom == "pressure_level":
            if request.zg_bottom_pressure is None:
                raise ValueError("zg_bottom_pressure must be specified when zg_bottom is 'pressure_level'.")

            p_min = float(level.min().values)
            p_max = float(level.max().values)
            if request.zg_bottom_pressure < p_min or request.zg_bottom_pressure > p_max:
                raise ValueError("Requested zg_bottom_pressure is outside dataset level range.")
            if request.zg_top_pressure < p_min or request.zg_top_pressure > p_max:
                raise ValueError("Requested zg_top_pressure is outside dataset level range.")
            if request.zg_top_pressure >= request.zg_bottom_pressure:
                raise ValueError("Requested zg_top_pressure must be less than zg_bottom_pressure.")

        # Determine cell-index range whose cells cover bbox
        # Cells i span [start[i], start[i+1]]
        i0 = int(np.searchsorted(lat_starts, lat0, side="right") - 1)
        i1 = int(np.searchsorted(lat_starts, lat1, side="left"))  # exclusive cell end index
        j0 = int(np.searchsorted(lon_starts, lon0, side="right") - 1)
        j1 = int(np.searchsorted(lon_starts, lon1, side="left"))

        # Clamp to valid cell index range
        i0 = max(i0, 0)
        j0 = max(j0, 0)
        i1 = min(i1, Nlat_cell)
        j1 = min(j1, Nlon_cell)

        # Apply margin in *cell indices*
        i0m = i0 + m
        i1m = i1 - m
        j0m = j0 + m
        j1m = j1 - m

        if i1m <= i0m or j1m <= j0m:
            raise ValueError("Domain too small after applying bbox + margin.")

        # Now select the corresponding *start indices* for those cells
        # We keep starts i0m..i1m-1 (length Ncells), and use start[i+1] for ends.
        lat_cell_slice = slice(i0m, i1m)  # cells
        lon_cell_slice = slice(j0m, j1m)

        out = ds.isel({lat_name: lat_cell_slice, lon_name: lon_cell_slice}).copy()

        lat_start = lat.isel({lat_name: lat_cell_slice}).values
        lat_end = lat.isel({lat_name: slice(i0m + 1, i1m + 1)}).values
        lat_mid = 0.5 * (lat_start + lat_end)

        lon_start = lon.isel({lon_name: lon_cell_slice}).values
        lon_end = lon.isel({lon_name: slice(j0m + 1, j1m + 1)}).values
        lon_mid = 0.5 * (lon_start + lon_end)

        # Vertical bounds unchanged
        p_edges = _cell_edges_from_centers(level, level_name)
        p_mid = np.asarray(level.values, dtype=float)

        lat_cell_id = np.arange(i0m, i0m + lat_mid.size, dtype=int)
        lon_cell_id = np.arange(j0m, j0m + lon_mid.size, dtype=int)
        p_cell_id = np.arange(p_mid.size, dtype=int)

        out = out.assign_coords(
            {
                lat_name: (lat_name, lat_mid.astype(np.float64)),
                lon_name: (lon_name, lon_mid.astype(np.float64)),
                "lat_cell_id": (lat_name, lat_cell_id),
                "lon_cell_id": (lon_name, lon_cell_id),
                "lat_start": (lat_name, lat_start.astype(np.float64)),
                "lat_end": (lat_name, lat_end.astype(np.float64)),
                "lon_start": (lon_name, lon_start.astype(np.float64)),
                "lon_end": (lon_name, lon_end.astype(np.float64)),
                "p_cell_id": (level_name, p_cell_id),
                "p_start": (level_name, p_edges[:-1].astype(np.float64)),
                "p_end": (level_name, p_edges[1:].astype(np.float64)),
                "p_mid": (level_name, p_mid.astype(np.float64)),
            }
        )

        out.attrs.update(
            {
                "lat_min": float(lat_start[0]),
                "lat_max": float(lat_end[-1]),
                "lon_min": float(lon_start[0]),
                "lon_max": float(lon_end[-1]),
                "margin": int(m),
                "horizontal_coord_type": "cell_center_with_bounds",
                "horizontal_input_interpretation": "cell_starts",
            }
        )

        return out


    ds_halo = _build_domain_for_margin(margin - 1)
    ds_domain = _build_domain_for_margin(margin)

    if eager_loading:
        ds_domain = ds_domain.assign(
            T=ds_domain["T"].persist(),
            w=ds_domain["w"].persist(),
            sp=ds_domain["sp"].persist(),
        )
        ds_halo = ds_halo.assign(
            sp=ds_halo["sp"].persist(),
        )
    
    
    spec = DomainSpec(
        lat_min=float(ds_domain.lat_start[0]),
        lat_max=float(ds_domain.lat_end[-1]),
        lon_min=float(ds_domain.lon_start[0]),
        lon_max=float(ds_domain.lon_end[-1]),
        zg_top_pressure=request.zg_top_pressure,
        zg_bottom=request.zg_bottom,
        zg_bottom_pressure=request.zg_bottom_pressure if request.zg_bottom == "pressure_level" else None,
    )

    return ds_domain, ds_halo, spec
'''

def crop_to_target_grid(ds: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """
    Align a dataset to the coordinates shared with a target dataset
    using nearest-neighbour selection.
    """
    indexers = {k: v for k, v in target.coords.items() if k in ds.dims}
    return ds.sel(indexers, method="nearest")


def get_boundary_line_elements(ds: xr.Dataset) -> xr.Dataset:
    """
    Return horizontal line elements for the four lateral walls.

    Returns
    -------
    xr.Dataset with:
      - dl_east(lat)   : meridional line element at lon = lon_max
      - dl_west(lat)   : meridional line element at lon = lon_min
      - dl_south(lon)  : zonal line element at lat = lat_min
      - dl_north(lon)  : zonal line element at lat = lat_max

    Units: m
    """
    for k in ("lat_start", "lat_end", "lon_start", "lon_end"):
        if k not in ds.coords:
            raise ValueError(f"Dataset must contain coordinate '{k}' from determine_domain().")

    lat_min = ds.attrs.get("lat_min")
    lat_max = ds.attrs.get("lat_max")
    lon_min = ds.attrs.get("lon_min")
    lon_max = ds.attrs.get("lon_max")
    if lat_min is None or lat_max is None or lon_min is None or lon_max is None:
        raise ValueError("Dataset must contain 'lat_min', 'lat_max', 'lon_min', 'lon_max' attributes.")

    dphi = np.abs(np.deg2rad(ds["lat_end"]) - np.deg2rad(ds["lat_start"]))   # (lat,)
    dy = (config.R_earth * dphi).rename("dl_east") # type: ignore
    dl_west = dy.rename("dl_west")

    dlon = np.abs(np.deg2rad(ds["lon_end"]) - np.deg2rad(ds["lon_start"]))   # (lon,)
    phi_south = np.deg2rad(float(lat_min))
    phi_north = np.deg2rad(float(lat_max))

    dl_south = (config.R_earth * np.cos(phi_south) * dlon).rename("dl_south")
    dl_north = (config.R_earth * np.cos(phi_north) * dlon).rename("dl_north")

    dl_east = dy.rename("dl_east")

    out = xr.Dataset({
        "dl_east": dl_east,
        "dl_west": dl_west,
        "dl_south": dl_south,
        "dl_north": dl_north,
    })

    for name, fixed_coord, fixed_value in [
        ("dl_east", "lon", float(lon_max)),
        ("dl_west", "lon", float(lon_min)),
        ("dl_south", "lat", float(lat_min)),
        ("dl_north", "lat", float(lat_max)),
    ]:
        out[name].attrs.update({
            "units": "m",
            "fixed_coord": fixed_coord,
            "fixed_value": fixed_value,
            "horizontal_coord_type": ds.attrs.get("horizontal_coord_type"),
            "horizontal_input_interpretation": ds.attrs.get("horizontal_input_interpretation"),
        })

    return out


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
    cell_area = cell_area.rename("A_horizontal") # type: ignore

    cell_area.attrs.update(
        {
            "units": "m2",
            "long_name": "Horizontal wall geometric area",
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
    # # dy(lat) = R * dphi
    # dphi = np.abs(np.deg2rad(ds["lat_end"]) - np.deg2rad(ds["lat_start"]))  # (lat,)
    # dy = config.R_earth * dphi  # (lat,)

    # # dx(lon) along constant-lat boundary, using each lon cell's dlon
    # dlon = np.abs(np.deg2rad(ds["lon_end"]) - np.deg2rad(ds["lon_start"]))  # (lon,)

    # phi_south = np.deg2rad(float(lat_min))
    # phi_north = np.deg2rad(float(lat_max))
    # dx_south = config.R_earth * np.cos(phi_south) * dlon  # (lon,)
    # dx_north = config.R_earth * np.cos(phi_north) * dlon  # (lon,)

    dl = get_boundary_line_elements(ds)

    east  = (dp * dl["dl_east"]).rename("east") # type: ignore
    west  = (dp * dl["dl_west"]).rename("west") # type: ignore
    south = (dp * dl["dl_south"]).rename("south") # type: ignore
    north = (dp * dl["dl_north"]).rename("north") # type: ignore

    east.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_max"]),
        "orientation": "meridional",
        "normal_convention": "geometric_only"   
    })

    west.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_min"]),
        "orientation": "meridional",
        "normal_convention": "geometric_only"
    })

    south.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_min"]),
        "orientation": "zonal",
        "normal_convention": "geometric_only"
    })

    north.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_max"]),
        "orientation": "zonal",
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

    out = xr.Dataset({"A_east": east, "A_west": west, "A_south": south, "A_north": north})

    # Carry through coordinate metadata so consumers can locate bounds easily
    # (They’re already in ds.coords; xarray will propagate coords for used dims.)
    out = out.assign_coords(
        {
            "p_start":   ds["p_start"],
            "p_end":     ds["p_end"],
            "p_mid":     ds.get("p_mid", ds["level"]),
            "lat_start": ds["lat_start"],
            "lat_end":   ds["lat_end"],
            "lon_start": ds["lon_start"],
            "lon_end":   ds["lon_end"],
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
    # ---- Vertical thickness (Pa): (level,) ----
    dp = np.abs(ds["p_end"] - ds["p_start"])
    
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
    cell_volume = (cell_area * dp).rename("V_cell")
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
            "dp":        dp,
            "p_start":   ds["p_start"],
            "p_end":     ds["p_end"],
            "lat_start": ds["lat_start"],
            "lat_end":   ds["lat_end"],
            "lon_start": ds["lon_start"],
            "lon_end":   ds["lon_end"],
            **({"lat_cell_id": ds["lat_cell_id"]} if "lat_cell_id" in ds.coords else {}),
            **({"lon_cell_id": ds["lon_cell_id"]} if "lon_cell_id" in ds.coords else {}),
        }
    )

    return cell_volume
