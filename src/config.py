"""
Docstring for eulerian_heat_budget.src.io

Defines:

- Physical constants (currently: g, R, cp)
- Standard pressure levels (e.g., LEVELS_HPA)
- Default dataset paths / region aliases (project-specific)
- Runtime defaults used by CLI (when flags are not provided)

Contains minimal/no runtime logic.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# input dataset paths
DEFAULT_DATA_SOURCE = "local_era5"

DEFAULT_LOCAL_PATH = "/home/mhpereir/downloads-mhpereir/ERA5_zg_PNW"
DEFAULT_ARCO_PATH  = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
DEFAULT_ARCO_TOKEN = "anon"
DEFAULT_ARCO_OPEN_MAX_ATTEMPTS: int = 4
DEFAULT_ARCO_OPEN_RETRY_BASE_DELAY_SECONDS: float = 15.0

DEFAULT_TIME_START = "1941-06-01T00:00:00" # start time for budget period
DEFAULT_TIME_END   = "1941-06-07T00:00:00" # end time for budget period 



# constants
g: float        = 9.806e0 #[m/s2] gravitational acceleration constant
R_value:float   = 2.870e2 #[J/(kg*K)] specific gas constant for dry air (R = R*/M, where R* is the universal gas constant and M is the molar mass of dry air)
R_earth: float  = 6.371e6 #Earth radius in meters
cp: float       = 1.005e3 #specific heat capacity of air in [J/(kg*K)]

# default config values
DEFAULT_BBOX          = (40, 60, -130, -110) # lat_min, lat_max, lon_min, lon_max for domain extent (before margin/snap)

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
DEFAULT_ZG_TOP_PA: float = 1 * 100 # upper boundary pressure for geopotential height budget in Pa

DEFAULT_PLOTS_OUTPUT:str = str(PROJECT_ROOT / "results" / "plots")
DEFAULT_DIAGNOSTIC_PLOTS: bool = False
DEFAULT_CONSTANT_TEMPERATURE_TEST: bool = False

n_time:int = 12
n_lat :int = 36
n_lon :int = 36

DEFAULT_CHUNKS_3D1 = {
    "time": n_time,  # 1 day per chunk
    "level": -1,     # keep full vertical column together
    "lat": n_lat,
    "lon": n_lon,
}
