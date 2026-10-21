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


class OvertureAmenities:
    """Nearest real businesses and parks around a site, from the Overture places file. Names and coordinates are as published; verify before print."""

    PICKS = {"retail": ("shopping", 4), "restaurant": ("food_and_drink", 4), "employer": ("services_and_business", 3), "health": ("health_care", 2), "park": (None, 2)}

    @staticmethod
    def around(site: Site, raw: Path, radius_km: float = 3.5) -> list[Amenity]:
        import pyarrow.parquet as pq
        from shapely import wkb
        t = pq.read_table(raw / "overture" / "places_houston.parquet", columns=["names", "taxonomy", "basic_category", "brand", "confidence", "operating_status", "geometry"]).to_pandas()
        t = t[(t["operating_status"] == "open") & (t["confidence"] >= 0.8)].copy()
        pts = [wkb.loads(b) for b in t["geometry"]]
        t["x"], t["y"] = zip(*[to_xy(p.x, p.y) for p in pts])
        sx, sy = site.xy
        t["d"] = np.hypot(t["x"] - sx, t["y"] - sy)
        t = t[t["d"] < radius_km * 1000]
        t["grp"] = t["taxonomy"].map(lambda x: x["hierarchy"][0] if x is not None and len(x["hierarchy"]) else None)
        t["prim"] = t["taxonomy"].map(lambda x: x["primary"] if x is not None else None)
        t["name"] = t["names"].map(lambda n: n["primary"] if n is not None else None)
        t["branded"] = t["brand"].notna()
        out, seen = [], set()
        for cat, (grp, n) in OvertureAmenities.PICKS.items():
            if cat == "park":
                sub = t[t["prim"].isin(["park", "dog_park", "playground", "nature_reserve"])]
            elif cat == "employer":
                sub = t[t["basic_category"].isin(["corporate_or_business_office"]) & t["branded"]]
            elif cat == "health":
                sub = t[t["basic_category"].isin(["hospital", "medical_center"])]
            else:
                sub = t[(t["grp"] == grp) & t["branded"]]
            sub = sub[sub["name"].notna()].sort_values("d")
            k = 0
            for _, r in sub.iterrows():
                key = r["name"].lower()
                if key in seen:
                    continue
                seen.add(key)
                lon, lat = to_lonlat(r["x"], r["y"])
                out.append(Amenity(r["name"], cat, lon, lat, None, f"Overture, confidence {r['confidence']:.2f}"))
                k += 1
                if k >= n:
                    break
        return out
