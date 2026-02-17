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
    # Compute local time tendency of temperature
    pass

def compute_advective_term(U, T): #A
    # Compute horizontal advection term
    pass

def compute_adiabatic_term(omega, T): # C
    # Compute adiabatic term (vertical motion)
    pass

def compute_diabatic_term(S, A, C): # D
    # Compute diabatic term as residual
    pass