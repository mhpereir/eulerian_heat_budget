"""Shared ARCO ERA5 variable maps for online staging and offline cache reads."""

from __future__ import annotations

from typing import Any


ARCO_CORE_VAR_MAP = {
    "temperature": "T",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "vertical_velocity": "w",
    "surface_pressure": "sp",
}

ARCO_ADVECTIVE_BENCHMARK_VAR_MAP = {
    "vertical_integral_of_eastward_heat_flux": "Fx_heat",
    "vertical_integral_of_northward_heat_flux": "Fy_heat",
    "vertical_integral_of_eastward_mass_flux": "Fx_mass",
    "vertical_integral_of_northward_mass_flux": "Fy_mass",
}

ARCO_COLUMN_BENCHMARK_VAR_MAP = {
    "vertical_integral_of_thermal_energy": "vithe",
    "vertical_integral_of_energy_conversion": "viec",
    "vertical_integral_of_divergence_of_thermal_energy_flux": "vithed",
    "vertical_integral_of_divergence_of_mass_flux": "vimad",
}

ARCO_BENCHMARK_VAR_MAP = {
    **ARCO_ADVECTIVE_BENCHMARK_VAR_MAP,
    **ARCO_COLUMN_BENCHMARK_VAR_MAP,
}

CORE_VAR_NAMES = tuple(ARCO_CORE_VAR_MAP.values())
ADVECTIVE_BENCHMARK_VAR_NAMES = tuple(ARCO_ADVECTIVE_BENCHMARK_VAR_MAP.values())
COLUMN_BENCHMARK_VAR_NAMES = tuple(ARCO_COLUMN_BENCHMARK_VAR_MAP.values())
BENCHMARK_VAR_NAMES = tuple(ARCO_BENCHMARK_VAR_MAP.values())

SURFACE_VARIABLE_ERROR = (
    "staged ARCO cache support for surface variables T2m, u10, and v10 is "
    "not implemented yet. Re-run without --use-surface-variables."
)


def require_no_surface_variables(surface_specs: Any) -> None:
    """Fail early for staged-cache v1 when optional surface fields are requested."""
    if bool(getattr(surface_specs, "use_surface_variables", False)):
        raise NotImplementedError(SURFACE_VARIABLE_ERROR)
