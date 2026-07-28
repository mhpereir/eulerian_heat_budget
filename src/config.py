"""
Docstring for eulerian_heat_budget.src.io

Defines:

- Physical constants (currently: g, R, cp)
- Standard pressure levels (e.g., LEVELS_HPA)
- Default dataset paths / region bbox aliases (project-specific)
- Runtime defaults used by CLI (when flags are not provided)

Contains minimal/no runtime logic.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# input dataset paths
DEFAULT_DATA_SOURCE = "local_era5"

DEFAULT_LOCAL_PATH: str | None = os.environ.get("EHB_DATA_ROOT") or None
DEFAULT_ARCO_PATH  = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_ARCO_TOKEN = "anon"
DEFAULT_ARCO_OPEN_MAX_ATTEMPTS: int = 4
DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS: float = 15.0
DEFAULT_ARCO_STAGE_ATTEMPT_TIMEOUT_SECONDS: float = 10800.0

DEFAULT_TIME_START = "1941-06-01T00:00:00" # start time for budget period
DEFAULT_TIME_END   = "1941-06-07T00:00:00" # end time for budget period 

# constants
g: float        = 9.806e0 #[m/s2] gravitational acceleration constant
R_value:float   = 2.870e2 #[J/(kg*K)] specific gas constant for dry air (R = R*/M, where R* is the universal gas constant and M is the molar mass of dry air)
R_earth: float  = 6.371e6 #Earth radius in meters
cp: float       = 1.005e3 #specific heat capacity of air in [J/(kg*K)]

# REGIONs (consistent with your threshold files)
# Values are (lat_min, lat_max, lon_min, lon_max).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "pnw_bartusek": (40, 60, -130, -110),
    "pnw_hotz":     (49, 59, -125, -115),
    "ocean_test": (25, 45, -170, -150),
    "eastern_canada": (42, 52, -83, -73),
    "alaska": (59.5, 69.5, -160, -150),
    "western_usa": (34, 44, -120, -110),
    "central_usa": (36, 46, -105, -95),
    "gulf_usa": (31, 41, -90, -80),
    "western_eu": (43, 53, -2, 8),
    "central_china": (25, 35, 105, 115),
}

DEFAULT_MARGIN_N: int = 1 # number of grid points to keep as margin when determining domain extent

DEFAULT_ZG_BOT_MODE = "surface_pressure" #"pressure_level" # or "surface_pressure"

# set True if "surface_pressure" is used as bottom boundary for geopotential height budget
# else False if "pressure_level"
DEFAULT_ALLOW_BOTTOM_OVERFLOW = True # allow bottom layer weights to exceed 1 if surface pressure exceeds grid bottom pressure (accounts for ps below grid bottom)

DEFAULT_USE_SURFACE_VARIABLES:bool = False # if True, include surface variables (T2m, u10, v10) in budget calculations; else use lowest model level variables
DEFAULT_SURFACE_VARIABLE_MODE = 'none' # 'none', 'combined', or 'diagnostic_only'; only relevant if DEFAULT_USE_SURFACE_VARIABLES is True


# pressure levels for geopotential height domain boundaries (in Pa)
# zg_bottom_pressure only used if zg_bottom == "pressure_level"; else surface pressure determines bottom boundary
DEFAULT_ZG_BOT_PA: float = 600 * 100 # lower boundary pressure for geopotential height budget in Pa
DEFAULT_ZG_TOP_PA: float = 700 * 100 # upper boundary pressure for geopotential height budget in Pa

DEFAULT_OUTPUT_ROOT: str = (
    os.environ.get("EHB_OUTPUT_ROOT") or str(PROJECT_ROOT / "results")
)
DEFAULT_DIAGNOSTIC_PLOTS: bool = False
DEFAULT_CONSTANT_TEMPERATURE_TEST: bool = False

# ERA5 vertically integrated benchmark fields cover the complete atmosphere.
# Restrict their use to the deepest pressure-level domain available from ERA5:
# 1 hPa to a lower boundary that follows surface pressure. The ERA5 diagnostic
# still includes the small atmospheric cap above 1 hPa.
FULL_COLUMN_BENCHMARK_TOP_PRESSURE_PA: float = 100.0

n_time:int = 12
n_lat :int = 36
n_lon :int = 36

DEFAULT_CHUNKS_3D1 = {
    "time": n_time,  # 1 day per chunk
    "level": -1,     # keep full vertical column together
    "lat": n_lat,
    "lon": n_lon,
}
