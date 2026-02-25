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

BotMode = Literal["surface_pressure", "pressure_level"]


@dataclass(frozen=True)
class DomainRequest:
    # User intent
    bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max) BEFORE margin/snap
    margin_n: int

    # Vertical intent (Pa)
    zg_top_pressure: float  
    zg_bottom: BotMode       
    zg_bottom_pressure: Optional[float]
    allow_bottom_overflow: bool
     

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
    allow_bottom_overflow: bool

    def validate(self) -> None:
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min > lat_max")
        if self.lon_min > self.lon_max:
            raise ValueError("lon_min > lon_max")
        if self.zg_bottom == "pressure_level" and self.zg_bottom_pressure is None:
            raise ValueError("zg_bottom_pressure must be set when zg_bottom='pressure_level'")
        if self.zg_bottom == "surface_pressure" and self.zg_bottom_pressure is not None:
            raise ValueError("zg_bottom_pressure must be None when zg_bottom='surface_pressure'")
