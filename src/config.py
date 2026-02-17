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