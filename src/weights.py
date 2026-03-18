'''
Docstring for eulerian_heat_budget.src.weights

Builds fractional occupancy weights that account for surfaces intersecting the volume.

Current responsibilities (implemented / in-progress):

- Volume occupancy weights on `(time, level, lat, lon)` that represent “fraction of the cell within the control volume”
- Area weights for control-volume faces (horizontal and vertical walls), with fractional masking where the surface intersects the face

These weights are the bridge between “geometry” and later “integrals/budget terms”.

Note: the detailed set of returned arrays and their naming should be documented in the schema section (Section 6) and treated as an API.
'''
import xarray as xr
import numpy as np

from .specs import DomainSpec

def area_weights_horizontal(ds: xr.Dataset, domain_spec: DomainSpec) -> xr.Dataset:
    """
    Binary weights for horizontal faces of the control volume.

    Top face (always considered):
      - W_top(time, lat, lon) = 1 if sp > zg_top_pressure else 0

    Bottom face (optional):
      - If config.zg_bottom == "pressure_level":
          W_bottom(time, lat, lon) = 1 if sp > zg_bottom_pressure else 0
      - If config.zg_bottom == "surface_pressure":
          bottom face weights are not returned (no surface-flux integration there)

    Notes
    -----
    - No fractional weights: pixels are either fully included or excluded.
    - Assumes ds["sp"] is in Pa and comparable to config.zg_*_pressure.
    """
    if "sp" not in ds:
        raise KeyError("area_weights_horizontal: ds must contain 'sp' (surface pressure).")
    if "time" not in ds["sp"].dims or "lat" not in ds["sp"].dims or "lon" not in ds["sp"].dims:
        raise ValueError("area_weights_horizontal: expected ds['sp'] dims to include ('time','lat','lon').")

    sp = ds["sp"].astype("float64")  # (time, lat, lon)

    p_top = float(domain_spec.zg_top_pressure)
    w_top = xr.where(sp > p_top, 1.0, 0.0).astype("float64").rename("W_top")
    w_top.attrs.update(
        long_name="Top horizontal face binary weight",
        description="1 if surface pressure is deeper than top face pressure (face in atmosphere); else 0 (face underground).",
        units="1",
        face="top",
        face_pressure_pa=p_top,
        bottom_boundary_mode=str(domain_spec.zg_bottom),
    )

    out_vars = {"W_top": w_top}

    # Bottom face only if it is a pressure-level boundary (internal CV face)
    if str(domain_spec.zg_bottom) == "pressure_level" and domain_spec.zg_bottom_pressure is not None:
        p_bot = float(domain_spec.zg_bottom_pressure) #type:ignore - guarded from in if-statement
        w_bottom = xr.where(sp > p_bot, 1.0, 0.0).astype("float64").rename("W_bottom")
        w_bottom.attrs.update(
            long_name="Bottom horizontal face binary weight",
            description="1 if surface pressure is deeper than bottom face pressure (face in atmosphere); else 0 (face underground).",
            units="1",
            face="bottom",
            face_pressure_pa=p_bot,
            bottom_boundary_mode=str(domain_spec.zg_bottom),
        )
        out_vars["W_bottom"] = w_bottom

    # If zg_bottom == "surface_pressure": intentionally do not return W_bottom
    return xr.Dataset(out_vars)

def area_weights_vertical(ds: xr.Dataset, domain_spec: DomainSpec) -> xr.Dataset:
    """
    Fractional area weights for the 4 vertical boundary walls (east/west/north/south),
    truncated by surface pressure.

    Returns
    -------
    xr.Dataset with variables:
      - east(time, level, lat)
      - west(time, level, lat)
      - south(time, level, lon)
      - north(time, level, lon)

    Weight definition (same as volume_weights, but evaluated on boundary lines)
    ------------------------------------------------------------------------
    For a layer with bounds [p_end, p_start] (p_start > p_end) and boundary surface pressure sp_b:

      raw = (sp_b - p_end) / (p_start - p_end)

      clip to [0,1] for all layers except optionally bottom layer (level index 0),
      which may exceed 1 if sp_b > p_start (accounts for sp below grid bottom).

    Assumptions
    -----------
    - level is descending in pressure and p_start(level) > p_end(level) for all levels
    - ds["sp"] has dims (time, lat, lon)
    - ds has been processed by determine_domain() so lat/lon represent cells
    """
    required = ["p_start", "p_end", "sp"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"area_weights_vertical: ds missing required variables/coords: {missing}")

    if "lat" not in ds.dims or "lon" not in ds.dims or "level" not in ds.dims:
        raise ValueError("area_weights_vertical: expected dims ('time','level','lat','lon') present in ds.")

    sp      = ds["sp"].astype("float64")     # (time, lat, lon)
    p_start = ds["p_start"].astype("float64") # (level,)
    p_end   = ds["p_end"].astype("float64")   # (level,)

    dp = p_start - p_end
    if bool((dp <= 0).any()):
        raise ValueError("area_weights_vertical: expected descending levels with p_start > p_end for all levels.")

    # select effective bottom pressure (either surface pressure; or fixed pressure level if zg_bottom is pressure_level)
    if str(domain_spec.zg_bottom) == "pressure_level": 
        #zg_bottom_pressure is a float (stric enforced by config schema) so safe to convert to DataArray with single value
        p_bot = xr.DataArray(float(domain_spec.zg_bottom_pressure), attrs={"units": "Pa"}) #type: ignore
        # If the surface is shallower than p_bot, the actual bottom is sp (no atmosphere below sp)
        p_bot_eff = xr.ufuncs.minimum(p_bot, sp)  # (time, lat, lon)
    elif str(domain_spec.zg_bottom) == "surface_pressure":
        p_bot_eff = sp  # (time, lat, lon)
    else:
        raise ValueError(f"area_weights_vertical: unsupported zg_bottom={domain_spec.zg_bottom}")

    # Bottom boundary pressure
    p_bot_eff_e = 0.5*(p_bot_eff.isel(lon=-1, lat=slice(1,-1)) + p_bot_eff.isel(lon=-2, lat=slice(1,-1)))  # (time, lat)  east boundary (lon_max)
    p_bot_eff_w = 0.5*(p_bot_eff.isel(lon=0, lat=slice(1,-1)) + p_bot_eff.isel(lon=1, lat=slice(1,-1)))    # (time, lat)  west boundary (lon_min)
    p_bot_eff_s = 0.5*(p_bot_eff.isel(lat=0, lon=slice(1,-1)) + p_bot_eff.isel(lat=1, lon=slice(1,-1)))    # (time, lon)  south boundary (lat_min)
    p_bot_eff_n = 0.5*(p_bot_eff.isel(lat=-1, lon=slice(1,-1)) + p_bot_eff.isel(lat=-2, lon=slice(1,-1)))  # (time, lon)  north boundary (lat_max)

    # Drop scalar coords that will conflict during Dataset merge
    p_bot_eff_e = _drop_if_present(p_bot_eff_e, ["lon", "lon_start", "lon_end", "lon_cell_id"])
    p_bot_eff_w = _drop_if_present(p_bot_eff_w, ["lon", "lon_start", "lon_end", "lon_cell_id"])
    p_bot_eff_s = _drop_if_present(p_bot_eff_s, ["lat", "lat_start", "lat_end", "lat_cell_id"])
    p_bot_eff_n = _drop_if_present(p_bot_eff_n, ["lat", "lat_start", "lat_end", "lat_cell_id"])

    # Raw fractional occupancies (broadcast dp/p_end over time+boundary-dim)
    # raw_e = (sp_e - p_end) / dp  # (time, level, lat)
    # raw_w = (sp_w - p_end) / dp  # (time, level, lat)
    # raw_s = (sp_s - p_end) / dp  # (time, level, lon)
    # raw_n = (sp_n - p_end) / dp  # (time, level, lon)

    w_e = _interval_overlap_fraction(p_start, p_end, domain_spec.zg_top_pressure, p_bot_eff_e)  # (time, level, lat)
    w_w = _interval_overlap_fraction(p_start, p_end, domain_spec.zg_top_pressure, p_bot_eff_w)  # (time, level, lat)
    w_s = _interval_overlap_fraction(p_start, p_end, domain_spec.zg_top_pressure, p_bot_eff_s)  # (time, level, lon)
    w_n = _interval_overlap_fraction(p_start, p_end, domain_spec.zg_top_pressure, p_bot_eff_n)  # (time, level, lon)

    # Default clipping
    # w_e = raw_e.clip(min=0.0, max=1.0)
    # w_w = raw_w.clip(min=0.0, max=1.0)
    # w_s = raw_s.clip(min=0.0, max=1.0)
    # w_n = raw_n.clip(min=0.0, max=1.0)

    if domain_spec.allow_bottom_overflow and str(domain_spec.zg_bottom) == "surface_pressure":
        # Bottom layer is assumed to be index 0 for descending pressure coordinates

        raw_e = (p_bot_eff_e - p_end) / dp  # (time, level, lat)
        raw_w = (p_bot_eff_w - p_end) / dp  # (time, level, lat)
        raw_s = (p_bot_eff_s - p_end) / dp  # (time, level, lon)
        raw_n = (p_bot_eff_n - p_end) / dp  # (time, level, lon)

        is_bottom = ds["level"] == ds["level"].isel(level=0)  # (level,) boolean

        w_e = xr.where(is_bottom, raw_e.isel(level=0).clip(min=0.0), w_e)
        w_w = xr.where(is_bottom, raw_w.isel(level=0).clip(min=0.0), w_w)
        w_s = xr.where(is_bottom, raw_s.isel(level=0).clip(min=0.0), w_s)
        w_n = xr.where(is_bottom, raw_n.isel(level=0).clip(min=0.0), w_n)


    w_e = w_e.transpose("time", "level", "lat")
    w_w = w_w.transpose("time", "level", "lat")
    w_s = w_s.transpose("time", "level", "lon")
    w_n = w_n.transpose("time", "level", "lon")

    w_east  = w_e.astype("float64").rename("W_east")
    w_west  = w_w.astype("float64").rename("W_west")
    w_south = w_s.astype("float64").rename("W_south")
    w_north = w_n.astype("float64").rename("W_north")

    w_east.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_max"]),
        "orientation": "meridional",
        "normal_convention": "geometric_only"   
    })

    w_west.attrs.update({
        "fixed_coord": "lon",
        "fixed_value": float(ds.attrs["lon_min"]),
        "orientation": "meridional",
        "normal_convention": "geometric_only"
    })

    w_south.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_min"]),
        "orientation": "zonal",
        "normal_convention": "geometric_only"
    })

    w_north.attrs.update({
        "fixed_coord": "lat",
        "fixed_value": float(ds.attrs["lat_max"]),
        "orientation": "zonal",
        "normal_convention": "geometric_only"
    })


    out = xr.Dataset({"W_east": w_east, "W_west": w_west, "W_south": w_south, "W_north": w_north})

    desc = (
        "Fractional wall occupancy above surface pressure; 0=underground, 1=in atmosphere, "
        + ("bottom layer may exceed 1 if sp > p_start" if domain_spec.allow_bottom_overflow else "clipped to [0,1]")
    )

    for da, ln in [
        (w_east,  "East vertical wall fractional weight (lon = lon_max)"),
        (w_west,  "West vertical wall fractional weight (lon = lon_min)"),
        (w_south, "South vertical wall fractional weight (lat = lat_min)"),
        (w_north, "North vertical wall fractional weight (lat = lat_max)"),
    ]:

        da.attrs.update({"long_name": ln, 
                         "description": desc,
                         "units": "1",
                         "horizontal_convention": ds.attrs.get("horizontal_coord_type"),
                         "horizontal_input_interpretation": ds.attrs.get("horizontal_input_interpretation"),
                         "vertical_convention": ds.attrs.get("vertical_coord_type"),
                         "vertical_input_interpretation": ds.attrs.get("vertical_input_interpretation"),
    })

    return out



def volume_weights(ds: xr.Dataset, domain_spec: DomainSpec) -> xr.DataArray:
    """
    Fractional volume weights for pressure layers truncated by surface pressure.

    Assumptions
    ---------------------
    - level is descending in pressure (downward): for each level, p_start > p_end
    - p_start(level), p_end(level) are layer bounds in Pa
    - surface pressure is ds["sp"] with dims (time, lat, lon), same units as p_start/p_end

    Meaning
    -------
    For each (time, level, lat, lon), weight w is the fraction of the layer [p_end, p_start]
    that is in-atmosphere (above ground), i.e. the portion with p <= sp.

      w = 0  if sp <= p_end   (surface above the layer: layer is entirely underground)
      w = 1  if sp >= p_start (surface below the layer: full layer exists)
      w = (sp - p_end) / (p_start - p_end) if p_end < sp < p_start

    Returns
    -------
    xr.Dataset with:
      - volume_weights(time, level, lat, lon) in [0, 1]
    """
    sp      = ds["sp"].astype("float64")      # (time, lat, lon)
    p_start = ds["p_start"].astype("float64")  # (level,)
    p_end   = ds["p_end"].astype("float64")    # (level,)

    dp    = p_start - p_end

    # select effective bottom pressure (either surface pressure; or fixed pressure level if zg_bottom is pressure_level)
    if str(domain_spec.zg_bottom) == "pressure_level": 
        #zg_bottom_pressure is a float (stric enforced by config schema) so safe to convert to DataArray with single value
        p_bot = xr.DataArray(float(domain_spec.zg_bottom_pressure), attrs={"units": "Pa"}) #type: ignore
        # If the surface is shallower than p_bot, the actual bottom is sp (no atmosphere below sp)
        p_bot_eff = xr.ufuncs.minimum(p_bot, sp)  # (time, lat, lon)
    elif str(domain_spec.zg_bottom) == "surface_pressure":
        p_bot_eff = sp  # (time, lat, lon)
    else:
        raise ValueError(f"volume_weights: unsupported zg_bottom={domain_spec.zg_bottom}")

    # Fractional occupancy above the surface
    w = _interval_overlap_fraction(p_start, p_end, domain_spec.zg_top_pressure, p_bot_eff).clip(min=0.0, max=1.0)

    w = w.transpose("time", "level", "lat", "lon")

    # w = raw_w.clip(min=0.0, max=1.0)
    
    if domain_spec.allow_bottom_overflow and str(domain_spec.zg_bottom) == "surface_pressure": 
        raw_w = ( (sp - p_end) / dp ).astype("float64") # (time, level, lat, lon)
        # allows for w > 1 if surface pressure is below (higher p) than the edge of the 
        # bottom layer
        w_bottom = raw_w.isel(level=0).clip(min=0.0)  # dims: (time, lat, lon)
        
        is_bottom = xr.zeros_like(w, dtype=bool)
        is_bottom[dict(level=0)] = True

        w = xr.where(is_bottom, w_bottom, w)


    out = w.rename("W_volume")

    out.attrs.update(
        long_name="Fractional layer volume above surface pressure",
        description=(
            "0=layer fully below ground; 1=layer fully in atmosphere; "
            "fractional if surface intersects layer; "
            + ("bottom layer may exceed 1 to account for sp > p_start" if domain_spec.allow_bottom_overflow else "")
        ),
        units="1",
        zg_top_pressure_pa=float(domain_spec.zg_top_pressure),
        zg_bottom_pressure_pa=float(domain_spec.zg_bottom_pressure) if str(domain_spec.zg_bottom) == "pressure_level" else None, #type: ignore
        zg_bottom_mode=str(domain_spec.zg_bottom),
    )

    out = out.assign_coords(
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
    return out


# ------------------------------------- Helper function --------------------------------------

def _drop_if_present(da: xr.DataArray, names: list[str]) -> xr.DataArray:
    present = [n for n in names if n in da.coords]
    return da.drop_vars(present, errors="ignore")


def _interval_overlap_fraction(p_hi, p_lo, p_top, p_bot):
    """
    Fraction of layer [p_lo, p_hi] that lies within [p_top, p_bot].
    All inputs broadcastable; requires p_hi > p_lo.
    """
    dp = p_hi - p_lo

    inc_lo = np.maximum(p_lo, p_top)
    inc_hi = np.minimum(p_hi, p_bot)
    inc_dp = np.maximum(0.0, inc_hi - inc_lo)
    return (inc_dp / dp).astype("float64")
