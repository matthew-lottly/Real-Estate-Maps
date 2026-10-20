from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle
from shapely.geometry import box

from ..composer import MapElement, RenderContext
from ..config import PLACES
from .rankings import Rankings


def halo(color: str = "white", w: float = 3.0):
    return [pe.withStroke(linewidth=w, foreground=color)]


class BaseLayers(MapElement):
    """Land (county polygons) on a water background."""

    name = "base"

    def draw(self, c: RenderContext) -> None:
        c.counties.plot(ax=c.ax, facecolor=c.theme.land, edgecolor=c.theme.county_line, linewidth=0.8, zorder=1)


class ChoroplethLayer(MapElement):
    """ZIP polygons filled by class; ZIPs without enough sales are hatched."""

    name = "choropleth"

    def draw(self, c: RenderContext) -> None:
        z = c.zips
        for i, col in enumerate(c.scheme.colors):
            sub = z[z["cls"] == i]
            if len(sub):
                sub.plot(ax=c.ax, facecolor=col, edgecolor=c.theme.zip_line, linewidth=0.55, zorder=2)
        nd = z[z["cls"] < 0]
        if len(nd):
            nd.plot(ax=c.ax, facecolor=c.theme.nodata, edgecolor=c.theme.nodata_hatch, hatch="////", linewidth=0.3, zorder=2)
            nd.boundary.plot(ax=c.ax, color=c.theme.zip_line, linewidth=0.55, zorder=2.1)


class CountyLabels(MapElement):
    """County names, placed by hand in lon/lat so they sit in open ground instead of on top of the ZIP detail."""

    name = "county_labels"
    AT = {"Harris": (-95.17, 30.02), "Montgomery": (-95.30, 30.21), "Waller": (-95.80, 30.06), "Liberty": (-94.92, 30.16),
          "Chambers": (-94.90, 29.93), "Fort Bend": (-95.80, 29.52), "Brazoria": (-95.40, 29.38), "Galveston": (-94.90, 29.40)}

    def draw(self, c: RenderContext) -> None:
        for name, (lon, lat) in self.AT.items():
            x, y = c.to_xy(lon, lat)
            c.ax.text(x, y, f"{name.upper()} COUNTY", ha="center", va="center", fontsize=8.5, color=c.theme.muted, alpha=0.6,
                      fontweight="bold", zorder=4, style="italic", path_effects=halo(c.theme.land, 2.5))


class PlaceLabels(MapElement):
    """Reference places with a white halo so labels stay readable on any fill."""

    name = "places"

    def draw(self, c: RenderContext) -> None:
        minx, miny, maxx, maxy = c.extent
        for name, lon, lat, rank in PLACES:
            x, y = c.to_xy(lon, lat)
            if not (minx < x < maxx and miny < y < maxy):
                continue
            big = rank == 1
            c.ax.plot(x, y, marker="o" if rank < 3 else "s" if "Airport" not in name else "P", markersize=7.5 if big else 5, mfc="white" if rank < 3 else c.theme.ink,
                      mec=c.theme.ink, mew=1.3, zorder=6)
            dx = 700 if "Airport" not in name else 700
            c.ax.text(x + dx, y + 450, name, fontsize=13 if big else 9.5 if rank == 2 else 7.6, fontweight="bold" if rank < 3 else "normal",
                      color=c.theme.ink if rank < 3 else c.theme.muted, ha="left", va="bottom", zorder=7, path_effects=halo(w=3.2),
                      style="normal" if rank < 3 else "italic")


class RankMarkers(MapElement):
    """Numbered badges on the highest (filled) and lowest (outlined) ZIPs; keyed to the ranking panel."""

    name = "rank_markers"

    def draw(self, c: RenderContext) -> None:
        top, bottom = Rankings.of(c)
        r = (c.extent[2] - c.extent[0]) * 0.0105
        for rows, filled in ((top, True), (bottom, False)):
            for row in rows:
                c.ax.add_patch(Circle((row.x, row.y), r, facecolor=c.theme.ink if filled else "white", edgecolor=c.theme.ink, linewidth=1.4, zorder=8))
                c.ax.text(row.x, row.y, str(row.rank), color="white" if filled else c.theme.ink, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=9)
