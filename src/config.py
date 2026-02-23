"""
Docstring for eulerian_heat_budget.src.io

Defines:

- Physical constants (currently: g, R, cp)
- Standard pressure levels (e.g., LEVELS_HPA)
- Default dataset paths / region aliases (project-specific)
- Runtime defaults used by CLI (when flags are not provided)

Contains minimal/no runtime logic.
"""

path_data = "/home/mhpereir/downloads-mhpereir/ERA5_zg_PNW"

# constants
g: float        = 9.806e0 #[m/s2] gravitational acceleration constant
R_earth: float  = 6.371e6 #Earth radius in meters
cp: float       = 1.005e3 #specific heat capacity of air in J/(kg*K)

# default config values
DEFAULT_BBOX          = (40, 60, -130, -110) # lat_min, lat_max, lon_min, lon_max for domain extent (before margin/snap)

DEFAULT_MARGIN_N: int = 0 # number of grid points to keep as margin when determining domain extent

DEFAULT_ZG_BOT_MODE = "pressure_level" # or "surface_pressure"

# set True if "surface_pressure" is used as bottom boundary for geopotential height budget
# else False if "pressure_level"
DEFAULT_ALLOW_BOTTOM_OVERFLOW = True # allow bottom layer weights to exceed 1 if surface pressure exceeds grid bottom pressure (accounts for ps below grid bottom)

# pressure levels for geopotential height domain boundaries (in Pa)
# zg_bottom_pressure only used if zg_bottom == "pressure_level"; else surface pressure determines bottom boundary
DEFAULT_ZG_BOT_PA: float = 500 * 100 # lower boundary pressure for geopotential height budget in Pa
DEFAULT_ZG_TOP_PA: float = 100 * 100 # upper boundary pressure for geopotential height budget in Pa