from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AcousticImportSettings:
    """Settings for linking bat acoustic detections to hedgerow segments."""

    source_format: str = "generic"
    hedge_id_column: str = "hf_uid"
    detection_hedge_id_column: str | None = None
    latitude_column: str | None = None
    longitude_column: str | None = None
    detections_crs: str = "EPSG:4326"
    max_distance_m: float | None = 50.0
    datetime_column: str | None = None
    species_column: str | None = None
    guild_column: str | None = None
    confidence_column: str | None = None
    activity_column: str | None = None
    min_confidence: float | None = None
    acoustic_timezone: str = "UTC"
    night_rollover_hour: int = 12
    unmatched_output: bool = False
