"""Validated, backend-neutral configuration for staged ARCO campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from src import config


SCHEMA_VERSION = 1
CAMPAIGN_FILE = "campaign.json"
CAMPAIGN_ID_PATTERN = re.compile(r"^[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class CampaignConfigError(ValueError):
    """Raised when campaign configuration is incomplete or inconsistent."""


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignConfigError(f"{field} must be a JSON object.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise CampaignConfigError(f"{field} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise CampaignConfigError(f"{field} contains unknown field(s): {', '.join(unknown)}")


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignConfigError(f"{field} must be an integer.")
    return value


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignConfigError(f"{field} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignConfigError(f"{field} must be finite.")
    return result


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CampaignConfigError(f"{field} must be true or false.")
    return value


def _parse_month_day(value: Any, field: str) -> tuple[int, int, str]:
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}-\d{2}", value):
        raise CampaignConfigError(f"{field} must use MM-DD format.")
    month, day = map(int, value.split("-"))
    try:
        date(2000, month, day)
    except ValueError as exc:
        raise CampaignConfigError(f"{field} is not a valid month-day: {value}") from exc
    return month, day, value


@dataclass(frozen=True)
class Campaign:
    campaign_id: str
    start_year: int
    end_year: int
    start_month_day: str
    end_month_day: str
    region: str | None
    bbox: tuple[float, float, float, float] | None
    margin_n: int
    zg_top_pa: float
    zg_bottom: str
    zg_bottom_pa: float | None
    allow_bottom_overflow: bool
    time_chunk: str
    attempt_timeout_seconds: float
    include_benchmark_variables: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "Campaign":
        config_path = Path(path)
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CampaignConfigError(f"Campaign file does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise CampaignConfigError(f"Campaign file is not valid JSON: {exc}") from exc
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Campaign":
        payload = _require_mapping(raw, "campaign")
        _require_exact_keys(
            payload,
            "campaign",
            {"campaign_id", "years", "season", "domain", "staging"},
            {"schema_version"},
        )
        schema_version = payload.get("schema_version", SCHEMA_VERSION)
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SCHEMA_VERSION
        ):
            raise CampaignConfigError(f"campaign.schema_version must be {SCHEMA_VERSION}.")

        campaign_id = payload["campaign_id"]
        if not isinstance(campaign_id, str) or not CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
            raise CampaignConfigError(
                "campaign_id must start with a lowercase letter, contain only "
                "lowercase letters, digits, and hyphens, end with a letter or "
                "digit, and have at most 63 characters."
            )

        years = _require_mapping(payload["years"], "years")
        _require_exact_keys(years, "years", {"start", "end"})
        start_year = _require_int(years["start"], "years.start")
        end_year = _require_int(years["end"], "years.end")
        if not 1 <= start_year <= end_year <= 9999:
            raise CampaignConfigError(
                "Campaign years must satisfy 1 <= years.start <= years.end <= 9999."
            )

        season = _require_mapping(payload["season"], "season")
        _require_exact_keys(season, "season", {"start_month_day", "end_month_day"})
        start_month, start_day, start_month_day = _parse_month_day(
            season["start_month_day"], "season.start_month_day"
        )
        end_month, end_day, end_month_day = _parse_month_day(
            season["end_month_day"], "season.end_month_day"
        )
        if (start_month, start_day) > (end_month, end_day):
            raise CampaignConfigError("Campaign seasons cannot cross a calendar-year boundary.")
        for year in range(start_year, end_year + 1):
            try:
                date(year, start_month, start_day)
                date(year, end_month, end_day)
            except ValueError as exc:
                raise CampaignConfigError(
                    f"Campaign season contains an invalid date for year {year}."
                ) from exc

        domain = _require_mapping(payload["domain"], "domain")
        _require_exact_keys(
            domain,
            "domain",
            {"margin_n", "zg_top_pa", "zg_bottom", "allow_bottom_overflow"},
            {"region", "bbox", "zg_bottom_pa"},
        )
        has_region = domain.get("region") is not None
        has_bbox = domain.get("bbox") is not None
        if has_region == has_bbox:
            raise CampaignConfigError("domain must provide exactly one of region or bbox.")

        region: str | None = None
        bbox: tuple[float, float, float, float] | None = None
        if has_region:
            region = domain["region"]
            if not isinstance(region, str) or region not in config.REGIONS:
                choices = ", ".join(sorted(config.REGIONS))
                raise CampaignConfigError(f"domain.region must be one of: {choices}")
        else:
            raw_bbox = _require_mapping(domain["bbox"], "domain.bbox")
            _require_exact_keys(
                raw_bbox,
                "domain.bbox",
                {"lat_min", "lat_max", "lon_min", "lon_max"},
            )
            lat_min = _require_number(raw_bbox["lat_min"], "domain.bbox.lat_min")
            lat_max = _require_number(raw_bbox["lat_max"], "domain.bbox.lat_max")
            lon_min = _require_number(raw_bbox["lon_min"], "domain.bbox.lon_min")
            lon_max = _require_number(raw_bbox["lon_max"], "domain.bbox.lon_max")
            if not -90 <= lat_min < lat_max <= 90:
                raise CampaignConfigError(
                    "domain.bbox latitude bounds must satisfy -90 <= lat_min < lat_max <= 90."
                )
            if not -180 <= lon_min < lon_max <= 360:
                raise CampaignConfigError(
                    "domain.bbox longitude bounds must satisfy -180 <= lon_min < lon_max <= 360."
                )
            bbox = (lat_min, lat_max, lon_min, lon_max)

        margin_n = _require_int(domain["margin_n"], "domain.margin_n")
        if margin_n < 1:
            raise CampaignConfigError("domain.margin_n must be at least one.")
        zg_top_pa = _require_number(domain["zg_top_pa"], "domain.zg_top_pa")
        if zg_top_pa <= 0:
            raise CampaignConfigError("domain.zg_top_pa must be positive.")
        zg_bottom = domain["zg_bottom"]
        if zg_bottom not in {"surface_pressure", "pressure_level"}:
            raise CampaignConfigError(
                "domain.zg_bottom must be surface_pressure or pressure_level."
            )
        zg_bottom_pa_raw = domain.get("zg_bottom_pa")
        if zg_bottom == "surface_pressure":
            if zg_bottom_pa_raw is not None:
                raise CampaignConfigError(
                    "domain.zg_bottom_pa must be omitted for surface_pressure."
                )
            zg_bottom_pa = None
        else:
            if zg_bottom_pa_raw is None:
                raise CampaignConfigError(
                    "domain.zg_bottom_pa is required for pressure_level."
                )
            zg_bottom_pa = _require_number(zg_bottom_pa_raw, "domain.zg_bottom_pa")
            if zg_bottom_pa <= zg_top_pa:
                raise CampaignConfigError(
                    "domain.zg_bottom_pa must be greater than domain.zg_top_pa."
                )
        allow_bottom_overflow = _require_bool(
            domain["allow_bottom_overflow"], "domain.allow_bottom_overflow"
        )

        staging = _require_mapping(payload["staging"], "staging")
        _require_exact_keys(
            staging,
            "staging",
            {"time_chunk", "attempt_timeout_seconds", "include_benchmark_variables"},
        )
        time_chunk = staging["time_chunk"]
        if time_chunk not in {"none", "day", "month"}:
            raise CampaignConfigError("staging.time_chunk must be none, day, or month.")
        attempt_timeout_seconds = _require_number(
            staging["attempt_timeout_seconds"], "staging.attempt_timeout_seconds"
        )
        if attempt_timeout_seconds < 0:
            raise CampaignConfigError(
                "staging.attempt_timeout_seconds cannot be negative."
            )
        include_benchmark_variables = _require_bool(
            staging["include_benchmark_variables"],
            "staging.include_benchmark_variables",
        )

        return cls(
            campaign_id=campaign_id,
            start_year=start_year,
            end_year=end_year,
            start_month_day=start_month_day,
            end_month_day=end_month_day,
            region=region,
            bbox=bbox,
            margin_n=margin_n,
            zg_top_pa=zg_top_pa,
            zg_bottom=zg_bottom,
            zg_bottom_pa=zg_bottom_pa,
            allow_bottom_overflow=allow_bottom_overflow,
            time_chunk=time_chunk,
            attempt_timeout_seconds=attempt_timeout_seconds,
            include_benchmark_variables=include_benchmark_variables,
        )

    @property
    def task_count(self) -> int:
        return self.end_year - self.start_year + 1

    def time_window(self, year: int) -> tuple[str, str]:
        if not self.start_year <= year <= self.end_year:
            raise CampaignConfigError(f"Year {year} is outside this campaign.")
        return (
            f"{year:04d}-{self.start_month_day}T00:00:00",
            f"{year:04d}-{self.end_month_day}T23:00:00",
        )

    def resolved_bbox(self) -> tuple[float, float, float, float]:
        if self.region is not None:
            return tuple(map(float, config.REGIONS[self.region]))
        assert self.bbox is not None
        return self.bbox

    def request_mapping(self) -> dict[str, Any]:
        return {
            "bbox": list(self.resolved_bbox()),
            "margin_n": self.margin_n,
            "zg_top_pressure": self.zg_top_pa,
            "zg_bottom": self.zg_bottom,
            "zg_bottom_pressure": self.zg_bottom_pa,
        }

    def to_mapping(self) -> dict[str, Any]:
        domain: dict[str, Any] = {
            "margin_n": self.margin_n,
            "zg_top_pa": self.zg_top_pa,
            "zg_bottom": self.zg_bottom,
            "allow_bottom_overflow": self.allow_bottom_overflow,
        }
        if self.region is not None:
            domain["region"] = self.region
        else:
            assert self.bbox is not None
            domain["bbox"] = {
                "lat_min": self.bbox[0],
                "lat_max": self.bbox[1],
                "lon_min": self.bbox[2],
                "lon_max": self.bbox[3],
            }
        if self.zg_bottom_pa is not None:
            domain["zg_bottom_pa"] = self.zg_bottom_pa
        return {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "years": {"start": self.start_year, "end": self.end_year},
            "season": {
                "start_month_day": self.start_month_day,
                "end_month_day": self.end_month_day,
            },
            "domain": domain,
            "staging": {
                "time_chunk": self.time_chunk,
                "attempt_timeout_seconds": self.attempt_timeout_seconds,
                "include_benchmark_variables": self.include_benchmark_variables,
            },
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def initialize_campaign(cache_root: str | Path, campaign: Campaign) -> Path:
    """Create or validate one immutable campaign descriptor atomically."""
    campaign = Campaign.from_mapping(campaign.to_mapping())
    root = Path(cache_root)
    if root.name != campaign.campaign_id:
        raise CampaignConfigError(
            f"Campaign cache root must end in {campaign.campaign_id!r}: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    path = root / CAMPAIGN_FILE
    if path.exists():
        existing = Campaign.from_file(path)
        if existing.canonical_json() != campaign.canonical_json():
            raise CampaignConfigError(
                f"Campaign root {root} is already bound to different configuration."
            )
        return path
    if (root / "cache.sqlite").exists() or (root / "tiles").exists():
        raise CampaignConfigError(
            f"Refusing to initialize campaign metadata over a legacy cache root: {root}"
        )

    descriptor, temp_name = tempfile.mkstemp(
        prefix=".campaign.",
        suffix=".tmp",
        dir=root,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(campaign.to_mapping(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            existing = Campaign.from_file(path)
            if existing.canonical_json() != campaign.canonical_json():
                raise CampaignConfigError(
                    f"Campaign root {root} was concurrently bound to different configuration."
                )
        return path
    finally:
        temp_path.unlink(missing_ok=True)
