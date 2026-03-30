"""
Docstring for eulerian_heat_budget.src.specs

Dataclasses that define the project's contracts:

- `DomainRequest`: user intent (bounds, margin, etc.)
- `DomainSpec`: resolved/validated domain (explicit bounds and metadata)

This is the home for “what the domain means” and should stay free of I/O and heavy computation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Dict, Any, Tuple

from . import config

BotMode    = Literal["surface_pressure", "pressure_level"]
SourceKind = Literal["local_era5", "arco_era5"]

@dataclass(frozen=True)
class DataSourceConfig:
    kind: SourceKind

    # local
    path_data: Optional[str] = None

    # ARCO
    arco_path: Optional[str] = None
    arco_storage_token: str = config.DEFAULT_ARCO_TOKEN
    chunks_time: int = config.n_time # number of time steps per chunk in ARCO dataset; used to optimize chunking for loading time slices

    # common selection
    time_start: Optional[str] = None
    time_end: Optional[str] = None

@dataclass(frozen=True)
class DomainRequest:
    # User intent
    bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max) BEFORE margin/snap
    margin_n: int

    # Vertical intent (Pa)
    zg_top_pressure: float  
    zg_bottom: BotMode       
    zg_bottom_pressure: Optional[float]

@dataclass(frozen=True)
class SurfaceBehaviour:
    # How to handle surface variables in the budget calculations
    allow_bottom_overflow: bool  # when surface pressure > lowest pressure level, allow weights >1 (True) or cap weights at 1 (False)
    use_surface_variables: bool  # whether to include surface variables (T2m, u10, v10) in budget calculations; else use lowest model level variables
    surface_variable_mode: Optional[Literal['none', 'combined', 'diagnostic_only']] 

@dataclass(frozen=True)
class DomainSpec:
    # Resolved bounds actually used
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    zg_top_pressure: float
    zg_bottom: BotMode
    zg_bottom_pressure: Optional[float]


    def validate(self) -> None:
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min > lat_max")
        if self.lon_min > self.lon_max:
            raise ValueError("lon_min > lon_max")
        if self.zg_bottom == "pressure_level" and self.zg_bottom_pressure is None:
            raise ValueError("zg_bottom_pressure must be set when zg_bottom='pressure_level'")
        if self.zg_bottom == "surface_pressure" and self.zg_bottom_pressure is not None:
            raise ValueError("zg_bottom_pressure must be None when zg_bottom='surface_pressure'")
    