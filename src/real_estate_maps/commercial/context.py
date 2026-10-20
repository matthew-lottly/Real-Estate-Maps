"""Base-map context (highways, cities, airports, Gulf) clipped from Natural Earth and saved small under data/processed/context."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import requests

from ..config import Paths

NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
LAYERS = ("ne_10m_roads", "ne_10m_populated_places_simple", "ne_10m_airports", "ne_10m_ocean", "ne_10m_lakes")
BBOX = (-97.6, 28.4, -93.2, 31.4)   # Houston region, WGS84


class ContextBuilder:
    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()

    @property
    def raw(self) -> Path:
        return self.p.raw / "ne"

    @property
    def out(self) -> Path:
        return self.p.processed / "context"

    def fetch(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        for name in LAYERS:
            dest = self.raw / f"{name}.geojson"
            if dest.exists():
                continue
            with requests.get(f"{NE}/{name}.geojson", stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)

    def build(self) -> dict[str, int]:
        self.out.mkdir(parents=True, exist_ok=True)
        sizes = {}
        for name in LAYERS:
            g = gpd.read_file(self.raw / f"{name}.geojson", bbox=BBOX)
            g = g.clip(BBOX) if name in ("ne_10m_ocean", "ne_10m_lakes") else g
            keep = {"ne_10m_roads": ["name", "type", "level", "expressway"], "ne_10m_populated_places_simple": ["name", "pop_max", "adm1name"],
                    "ne_10m_airports": ["name", "iata_code", "type"], "ne_10m_ocean": [], "ne_10m_lakes": ["name"]}[name]
            g = g[[c for c in keep if c in g.columns] + ["geometry"]]
            g.to_file(self.out / f"{name.replace('ne_10m_', '')}.geojson", driver="GeoJSON")
            sizes[name] = len(g)
        return sizes
