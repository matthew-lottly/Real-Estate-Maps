"""What the broker supplies: a site (address + parcel), an amenity list and optional demographics."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Point, Polygon, shape

CRS = "EPSG:3081"
_to_xy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform
_to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
M_PER_MI, M_PER_FT, SQM_PER_ACRE = 1609.344, 0.3048, 4046.8564224


def to_xy(lon: float, lat: float) -> tuple[float, float]:
    return _to_xy(lon, lat)


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    return _to_ll(x, y)


@dataclass
class Site:
    name: str
    address: str
    lon: float
    lat: float
    parcel: Polygon | None = None          # projected (EPSG:3081); None -> no boundary supplied
    callouts: list[str] = field(default_factory=list)
    sample: bool = False

    @property
    def xy(self) -> tuple[float, float]:
        return to_xy(self.lon, self.lat)

    @property
    def acres(self) -> float | None:
        return self.parcel.area / SQM_PER_ACRE if self.parcel is not None else None

    @classmethod
    def from_geojson(cls, name: str, address: str, parcel_geojson: Path, callouts: list[str] | None = None) -> "Site":
        """Real client input: the parcel boundary as a GeoJSON polygon in WGS84."""
        g = gpd.read_file(parcel_geojson).to_crs(CRS)
        poly = g.geometry.union_all().convex_hull if len(g) > 1 else g.geometry.iloc[0]
        lon, lat = to_lonlat(*poly.centroid.coords[0])
        return cls(name, address, lon, lat, poly, callouts or [])


class SampleSite:
    """Illustrative stand-in. NOT a real tax lot: a rotated rectangle placed near the Galleria area for the demo maps."""

    @staticmethod
    def make() -> Site:
        lon, lat = -95.4905, 29.7545
        cx, cy = to_xy(lon, lat)
        w, d = 330 * M_PER_FT, 470 * M_PER_FT
        poly = affinity.rotate(Polygon([(cx - w / 2, cy - d / 2), (cx + w / 2, cy - d / 2), (cx + w / 2, cy + d / 2), (cx - w / 2, cy + d / 2)]), 17, origin=(cx, cy))
        return Site("Sample Site", "Sample address, Houston, TX (illustrative)", lon, lat, poly,
                    ["Highway frontage", "Signalized corner", "Dock-high loading", "Surface parking"], sample=True)


@dataclass
class Amenity:
    name: str
    category: str
    lon: float
    lat: float
    logo: Path | None = None
    note: str = ""

    @property
    def xy(self):
        return to_xy(self.lon, self.lat)


class AmenityList:
    """CSV with columns: name,category,lon,lat[,logo][,note]. `logo` is a PNG filename next to the CSV; without one a monogram badge is drawn."""

    @staticmethod
    def load(path: Path) -> list[Amenity]:
        out = []
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                logo = Path(path).parent / "logos" / r["logo"] if r.get("logo") else None
                out.append(Amenity(r["name"].strip(), r["category"].strip().lower(), float(r["lon"]), float(r["lat"]),
                                   logo if logo and logo.exists() else None, r.get("note", "")))
        return out


class DemographicProvider:
    """Optional CSV (ring_miles,metric,value[,label]) from the client's vendor/ACS extract. Missing file -> no demographic callouts."""

    def __init__(self, path: Path | None = None):
        self.rows: dict[int, dict[str, str]] = {}
        if path and Path(path).exists():
            with open(path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.rows.setdefault(int(float(r["ring_miles"])), {})[r["metric"]] = r["value"]

    @property
    def available(self) -> bool:
        return bool(self.rows)

    def get(self, ring: int) -> dict[str, str]:
        return self.rows.get(ring, {})
