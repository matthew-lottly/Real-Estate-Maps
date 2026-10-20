"""Paths, the Houston map frame and shared constants."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def market_csv(self) -> Path:
        return self.processed / "houston_zip_market.csv"

    @property
    def zip_geojson(self) -> Path:
        return self.processed / "houston_zip_boundaries.geojson"

    @property
    def counties_geojson(self) -> Path:
        return self.processed / "texas_counties.geojson"


@dataclass(frozen=True)
class HoustonFrame:
    """The mapped area (WGS84 lon/lat box) and the projection used for drawing (Texas Centric Albers, metres)."""

    west: float = -95.98
    south: float = 29.30
    east: float = -94.78
    north: float = 30.26
    crs: str = "EPSG:3081"
    metro_name: str = "Houston-The Woodlands-Sugar Land"


# Reference places for labels: (name, lon, lat, rank 1=big city 2=town 3=landmark). Approximate public coordinates.
PLACES: list[tuple[str, float, float, int]] = [
    ("Downtown Houston", -95.3698, 29.7604, 1), ("The Woodlands", -95.4613, 30.1658, 2), ("Katy", -95.8244, 29.7858, 2),
    ("Sugar Land", -95.6349, 29.5994, 2), ("Pearland", -95.2860, 29.5636, 2), ("Baytown", -94.9774, 29.7355, 2),
    ("Cypress", -95.6972, 29.9691, 3), ("Spring", -95.4172, 30.0799, 3), ("Clear Lake", -95.0958, 29.5558, 3),
    ("Galleria", -95.4618, 29.7387, 3), ("Humble", -95.2622, 29.9988, 3), ("Missouri City", -95.5377, 29.6186, 3),
    ("George Bush Intercontinental Airport", -95.3368, 29.9902, 3), ("William P. Hobby Airport", -95.2789, 29.6454, 3),
]
