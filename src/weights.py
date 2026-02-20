'''
Docstring for eulerian_heat_budget.src.weights

Constructs:

- `volume_weights(time,p,y,x)`
- `area_weights_vertical(dim_x, p)`
- `area_weights_horizontal(lat, lon)`

Ensures correct truncation at surface pressure.
'''
import xarray as xr

def volume_weights(ds: xr.Dataset, allow_bottom_overflow: bool = True) -> xr.Dataset:
    """
    Fractional volume weights for pressure layers truncated by surface pressure.

    Assumptions
    ---------------------
    - level is descending in pressure (downward): for each level, p_start > p_end
    - p_start(level), p_end(level) are layer bounds in Pa
    - surface pressure is ds["sfp"] with dims (time, lat, lon), same units as p_start/p_end

    Meaning
    -------
    For each (time, level, lat, lon), weight w is the fraction of the layer [p_end, p_start]
    that is in-atmosphere (above ground), i.e. the portion with p <= ps.

      w = 0  if ps <= p_end   (surface above the layer: layer is entirely underground)
      w = 1  if ps >= p_start (surface below the layer: full layer exists)
      w = (ps - p_end) / (p_start - p_end) if p_end < ps < p_start

    Returns
    -------
    xr.Dataset with:
      - volume_weights(time, level, lat, lon) in [0, 1]
    """
    required = ["p_start", "p_end", "sfp"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"volume_weights: ds missing required variables/coords: {missing}")

    ps      = ds["sfp"].astype("float64")      # (time, lat, lon)
    p_start = ds["p_start"].astype("float64")  # (level,)
    p_end   = ds["p_end"].astype("float64")    # (level,)

    # Ensure broadcasting works cleanly
    # xarray will broadcast (level,) against (time,lat,lon) automatically in arithmetic.
    dp = p_start - p_end

    # Guard: dp must be positive if p_start > p_end (descending levels)
    # (if dp<=0 anywhere, something is inconsistent with the "descending" assumption)
    if bool((dp <= 0).any()):
        bad = dp.where(dp <= 0, drop=True)
        raise ValueError(
            "volume_weights: expected descending levels with p_start > p_end for all levels, "
            "but found non-positive thickness (p_start - p_end) at some levels."
        )

    raw_w = ( (ps - p_end) / dp ).astype("float64") # (time, level, lat, lon)

    # Fractional occupancy above the surface
    w = raw_w.clip(min=0.0, max=1.0)

    if allow_bottom_overflow: 
        # allows for w > 1 if surface pressure is below (higher p) than the edge of the 
        # bottom layer
        w_bottom = raw_w.isel(level=0).clip(min=0.0)  # dims: (time, lat, lon)
        w[dict(level=0)] = w_bottom

    out = xr.Dataset({"volume_weights": w})
    out["volume_weights"].attrs.update(
        long_name="Fractional layer occupancy above surface pressure",
        description=(
            "0=layer fully below ground; 1=layer fully in atmosphere; "
            "fractional if surface intersects layer; "
            + ("bottom layer may exceed 1 to account for ps > p_start" if allow_bottom_overflow else "")
        ),
        units="1",
    )
    return out


def area_weights_vertical(ds):
    # Construct 2D vertical weights (pressure coordinates) for valid grid cells (e.g., atmosphere vs below surface)
    # horizontal dimension is either lat or lon, and vertical dimension is pressure
    # In case lat, area must account for spherical geometry, and in case of lon, doesn't matter.

    pass

def area_weights_horizontal(ds):
    # Construct 2D horizontal weights for valid grid cells (e.g., atmosphere vs below surface)
    # horizontal dimensions are both lat/lon, no vertical
    # area must account for spherical geometry 
    pass