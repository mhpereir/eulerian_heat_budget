'''
Docstring for eulerian_heat_budget.src.terms

Computes individual budget components:

- `compute_storage(T)`
- `compute_advective_term(U, T)`
- `compute_adiabatic_term(omega, T)`
- `compute_diabatic_term(S,A,C)` #residual term 

May call `integrals.py` internally.
'''

def compute_storage(T): #S
    r'''
    Inputs: T (temperature field, 4D: time, level, lat, lon)
    Outputs: S (storage term, 3D: time-2)
    
    math term: d/dt \int T dV
    '''
    # Compute local time tendency of temperature
    
    pass

def compute_advective_term(U, T, DomainSpec): #A
    r'''
    math term: -\int T U \cdot dA
    '''
    # Compute horizontal/vertical walls advection terms (top, east, west, south, north)
    omega_top = U['omega'].sel(level=DomainSpec.zg_top_pressure)  # vertical velocity at top wall

    if DomainSpec.zg_bottom == "surface_pressure":
        omega_bottom = None
    elif DomainSpec.zg_bottom == "pressure_level":
        omega_bottom = U['omega'].sel(level=DomainSpec.zg_bottom_pressure)  # vertical velocity at bottom wall (fixed pressure)
    else:
        raise ValueError(f"Invalid zg_bottom mode: {DomainSpec.zg_bottom}")

    pass

def compute_adiabatic_term(omega, T): # C
    r'''
    math term: -\int \omega * dT/dp dV 
    '''
    # Compute adiabatic term (vertical motion)
    pass

def compute_diabatic_term(S, A, C): # D
    r'''
    math term: D = S - A - C
    '''
    # Compute diabatic term as residual
    return S - A - C