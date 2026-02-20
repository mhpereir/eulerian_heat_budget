'''
Docstring for eulerian_heat_budget.src.io

Defines:

- Physical constants:
  - `g`
  - `R`
  - `cp`
- Upper boundary pressure
- Region configuration
- Default integration methods
'''

g: float        = 9.81    #[m/s2] gravitational acceleration constant
R_earth: float  = 6.371e6 #Earth radius in meters
cp: float       = 3.985e3 #specific heat capacity of seawater in J/(kg*K)

margin: int     = 1       # number of grid points to keep as margin when determining domain extent

zg_bottom:str = "pressure_level" # or "surface_pressure"

# pressure levels for geopotential height domain boundaries (in Pa)
# zg_bottom_pressure only used if zg_bottom == "pressure_level"; else surface pressure determines bottom boundary
zg_bottom_pressure: float = 1000 * 100 # lower boundary pressure for geopotential height budget in Pa
zg_top_pressure: float    = 500 * 100 # upper boundary pressure for geopotential height budget in Pa
