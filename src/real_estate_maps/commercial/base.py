"""Shared inputs and drawing helpers for every commercial map."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ..config import PLACES, Paths
from ..data import MarketDataset
from .brand import Brand
from .context import ContextBuilder
from .inputs import CRS, M_PER_MI, AmenityList, DemographicProvider, SampleSite, Site, to_xy
from .page import Page, halo
from .providers import DriveTimeProvider, ImageryProvider, NoDriveTime, NoImagery


@dataclass
class MapContext:
    brand: Brand
    site: Site
    amenities: list = field(default_factory=list)
    demographics: DemographicProvider = field(default_factory=DemographicProvider)
    imagery: ImageryProvider = field(default_factory=NoImagery)
    drive_time: DriveTimeProvider = field(default_factory=NoDriveTime)
    paths: Paths = field(default_factory=Paths)

    def __post_init__(self):
        c = ContextBuilder(self.paths).out
        self.roads = gpd.read_file(c / "roads.geojson").to_crs(CRS)
        self.places = gpd.read_file(c / "populated_places_simple.geojson").to_crs(CRS)
        self.airports = gpd.read_file(c / "airports.geojson").to_crs(CRS)
        self.ocean = gpd.read_file(c / "ocean.geojson").to_crs(CRS)
        self.lakes = gpd.read_file(c / "lakes.geojson").to_crs(CRS)
        self.dataset = MarketDataset(self.paths)
        self.zips = self.dataset.snapshot().to_crs(CRS)
        self.counties = self.dataset.counties.to_crs(CRS)

    @property
    def stamp(self) -> str | None:
        bits = [("SAMPLE SITE" if self.site.sample else None), ("SAMPLE BRAND KIT" if self.brand.sample else None)]
        bits = [b for b in bits if b]
        return " · ".join(bits) + " — illustrative" if bits else None


class RoadStyle:
    """Highway classes drawn from Natural Earth (generalized, majors only; local streets are not in this source)."""

    @staticmethod
    def label(r) -> str | None:
        n = r["name"]
        if not isinstance(n, str) or not n:
            return None
        if r["type"] == "Beltway":
            return f"Beltway {n}"
        return {"Interstate": f"I-{n}", "Federal": f"US {n}", "State": f"TX {n}"}.get(r["level"], f"FM {n}" if r["level"] == "Other" else None)

    @staticmethod
    def rank(r) -> int:
        if r["type"] == "Beltway" or r["level"] == "Interstate":
            return 0
        return 1 if r["level"] == "Federal" else 2


class BaseMap:
    """Water, land and roads, shared by the maps that need geography."""

    def __init__(self, c: MapContext, page: Page):
        self.c, self.p = c, page

    def water_land(self, ax, counties: bool = True) -> None:
        b = self.c.brand
        with self.p.layer("basemap-water-land", ax):
            ax.set_facecolor(b.land)
            self.c.ocean.plot(ax=ax, fc=b.water, ec="none", zorder=1)
            self.c.lakes.plot(ax=ax, fc=b.water, ec="none", zorder=1)
            if counties:
                self.c.counties.boundary.plot(ax=ax, color="#C9C3B6", lw=0.5, ls="--", zorder=1.5)

    def streets(self, ax, label: bool = True, light: bool = False) -> None:
        """Real streets (Overture Maps), detail chosen from the map scale; `light` keeps only the main roads for use under a choropleth."""
        from .streets import OvertureStreets
        OvertureStreets(self.c.paths.raw).draw(self.p, ax, self.c.brand, label, light)

    def outlines(self, ax) -> None:
        """Local-scale context: ZIP outlines. (The only road source here is generalized highways, which would mislead at this zoom.)"""
        with self.p.layer("zip-outlines", ax):
            self.c.zips.boundary.plot(ax=ax, color="#C4BDAE", lw=0.6, zorder=2)

    def roads(self, ax, label: bool = True, scale: float = 1.0) -> None:
        b, r = self.c.brand, self.c.roads.copy()
        r["rank"] = r.apply(RoadStyle.rank, axis=1)
        r["lab"] = r.apply(RoadStyle.label, axis=1)
        spec = {2: (0.9, b.muted, 1.0), 1: (1.5, b.secondary, 1.6), 0: (2.4, b.primary, 2.8)}
        with self.p.layer("roads", ax):
            for rk in (2, 1, 0):
                sub = r[r["rank"] == rk]
                lw, col, case = spec[rk]
                sub.plot(ax=ax, color="white", lw=(lw + 1.6) * scale, zorder=3 + rk * 0.1, capstyle="round")
                sub.plot(ax=ax, color=col, lw=lw * scale, zorder=3.05 + rk * 0.1, capstyle="round")
        if not label:
            return
        x0, y0, x1, y1 = self.p.extent(ax)
        from shapely.geometry import box
        view = box(x0, y0, x1, y1)
        placed: list[tuple[float, float]] = []
        W = x1 - x0
        with self.p.layer("road-labels", ax):
            for lab, grp in r[r["lab"].notna() & (r["rank"] < 2)].groupby("lab"):
                best = None
                for g in grp.geometry:
                    part = g.intersection(view)
                    if part.is_empty:
                        continue
                    if part.geom_type == "MultiLineString":
                        part = max(part.geoms, key=lambda s: s.length)
                    if part.geom_type != "LineString":
                        continue
                    if best is None or part.length > best.length:
                        best = part
                if best is None or best.length < W * 0.06:
                    continue
                p = best.interpolate(0.5, normalized=True)
                if any(np.hypot(p.x - q[0], p.y - q[1]) < W * 0.07 for q in placed):
                    continue
                placed.append((p.x, p.y))
                ax.text(p.x, p.y, lab, fontsize=6.2, fontweight="bold", color="white", ha="center", va="center", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.22", fc=b.primary if grp["rank"].min() == 0 else b.secondary, ec="white", lw=0.8))

    def highway_legend(self, a, x, y) -> None:
        b = self.c.brand
        for i, (lw, col, name) in enumerate([(2.4, b.primary, "Interstate / Beltway"), (1.5, b.secondary, "U.S. highway"), (0.9, b.muted, "State highway")]):
            a.plot([x, x + 0.08], [y - i * 0.032] * 2, color=col, lw=lw, transform=a.transAxes, solid_capstyle="round")
            a.text(x + 0.11, y - i * 0.032, name, fontsize=6.6, va="center", color=b.ink, transform=a.transAxes)


class CommercialMap(abc.ABC):
    key = "map"
    title = ""

    def __init__(self, c: MapContext):
        self.c = c

    @abc.abstractmethod
    def build(self) -> Page: ...

    def export(self, outdir: Path) -> dict[str, Path]:
        pg = self.build()
        out = pg.export(outdir, f"{self.c.site.name.lower().replace(' ', '_')}__{self.key}")
        out["issues"] = pg.issues
        return out


def miles_between(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]) / M_PER_MI)


def destinations(ctx: MapContext) -> list[tuple[str, tuple[float, float]]]:
    """Named reference destinations for distance tables (approximate public coordinates)."""
    want = ["Downtown Houston", "George Bush Intercontinental Airport", "William P. Hobby Airport", "The Woodlands", "Sugar Land", "Katy", "Baytown", "Pearland"]
    ll = {n: (lo, la) for n, lo, la, _ in PLACES}
    out = [(n, to_xy(*ll[n])) for n in want if n in ll]
    g = ctx.places[ctx.places["name"] == "Galveston"]
    if len(g):
        out.append(("Galveston", (g.geometry.iloc[0].x, g.geometry.iloc[0].y)))
    return out
