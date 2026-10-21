from __future__ import annotations

import geopandas as gpd
import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.ops import unary_union

from .base import BaseMap, CommercialMap
from .page import Page, halo

SRC = ("Streets: Overture Maps (release 2026-09-23.1); water and the inset highways: Natural Earth, generalized; state outline: public county boundaries. "
       "Frames show the extent of the next map in the series.")


class SiteLocatorMap(CommercialMap):
    """Three scales on one page: the market area, the metro and the state, so a reader can place the site at a glance."""

    key = "site_locator"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        pg = Page(b, "Site Location", f"{c.site.name}  ·  {c.site.address}", SRC, c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), c.site.xy, 36_000)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        base.streets(ax)
        x0, y0, x1, y1 = pg.extent(ax)
        with pg.layer("city-labels", ax):
            for _, r in c.places.iterrows():
                if x0 < r.geometry.x < x1 and y0 < r.geometry.y < y1 and r["pop_max"] > 100000:
                    ax.annotate(r["name"].upper(), (r.geometry.x, r.geometry.y), xytext=(0, 0), textcoords="offset points", ha="center", fontsize=9, color=b.muted, fontweight="bold", alpha=0.8, path_effects=halo())
        pg.site_pin(ax, c.site.xy, c.site.name, dx=10, dy=-18)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        tx = unary_union(list(c.counties.geometry))
        boxes = [("Houston metro", (c.site.xy[0], c.site.xy[1]), 150_000, (0.69, 0.47, 0.28, 0.43), (x0, y0, x1, y1)),
                 ("Texas", None, None, (0.69, 0.07, 0.28, 0.36), None)]
        # metro inset
        name, ctr, wid, rect, frame = boxes[0]
        panel = pg.fig.add_axes(rect)
        h = wid * rect[3] * 8.5 / (rect[2] * 11)
        panel.set_xlim(ctr[0] - wid / 2, ctr[0] + wid / 2)
        panel.set_ylim(ctr[1] - h / 2, ctr[1] + h / 2)
        panel.set_aspect("equal")
        panel.set_xticks([]); panel.set_yticks([])
        panel.set_facecolor(b.land)
        c.ocean.plot(ax=panel, fc=b.water, ec="none")
        c.counties.boundary.plot(ax=panel, color="#C9C3B6", lw=0.5)
        BaseMap(c, pg).roads(panel, label=False, scale=0.55)
        panel.add_patch(Rectangle((frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1], fc="none", ec=b.alert, lw=1.4, zorder=20))
        panel.plot(*c.site.xy, marker="*", ms=9, mfc=b.alert, mec="white", zorder=21)
        panel.text(0.05, 0.94, "HOUSTON METRO", transform=panel.transAxes, fontsize=6.5, fontweight="bold", color=b.primary, va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        for s in panel.spines.values():
            s.set_edgecolor(b.primary)
        # Texas inset
        t = pg.fig.add_axes(boxes[1][3])
        gpd_t = gpd.GeoSeries([tx.buffer(1500).buffer(-1500)], crs=c.counties.crs)
        gpd_t.plot(ax=t, fc=b.land, ec=b.primary, lw=0.8)
        t.add_patch(Rectangle((c.site.xy[0] - 75_000, c.site.xy[1] - 75_000), 150_000, 150_000, fc="none", ec=b.alert, lw=1.4))
        t.plot(*c.site.xy, marker="*", ms=7, mfc=b.alert, mec="white")
        t.set_aspect("equal"); t.set_axis_off()
        t.text(0.02, 0.98, "TEXAS", transform=t.transAxes, fontsize=6.5, fontweight="bold", color=b.primary, va="top")
        return pg
