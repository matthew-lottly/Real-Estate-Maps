from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle

from .base import BaseMap, CommercialMap, destinations, miles_between
from .inputs import M_PER_MI
from .page import Page, halo

SRC = ("Highways, cities, airports and water: Natural Earth (public domain), generalized — major highways only. Reference destinations: approximate public coordinates. "
       "Distances are straight-line; drive times need a routing service, which is not connected.")


class RegionalAccessMap(CommercialMap):
    key = "regional_access"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        pg = Page(b, "Regional Access", f"{c.site.name}  ·  {c.site.address}", SRC, c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), c.site.xy, 140_000)
        base = BaseMap(c, pg)
        base.water_land(ax)
        base.roads(ax)
        x0, y0, x1, y1 = pg.extent(ax)
        sx, sy = c.site.xy
        with pg.layer("distance-rings", ax):
            for mi in (10, 25, 50):
                ax.add_patch(Circle((sx, sy), mi * M_PER_MI, fc="none", ec=b.primary, lw=0.8, ls=(0, (5, 4)), alpha=0.7, zorder=5))
                ax.text(sx + mi * M_PER_MI * 0.7071, sy - mi * M_PER_MI * 0.7071, f"{mi} mi", fontsize=6.5, color=b.primary, fontweight="bold", ha="left", va="top", zorder=6, path_effects=halo())
        dests = destinations(c)
        with pg.layer("cities-airports", ax):
            for n, xy in dests:
                air = "Airport" in n
                ax.plot(*xy, marker="P" if air else "o", ms=5.5 if air else 4.5, mfc=b.ink if air else "white", mec=b.ink, mew=1, zorder=14)
                short = n.replace("George Bush Intercontinental Airport", "IAH Airport").replace("William P. Hobby Airport", "Hobby Airport")
                edge = xy[0] > x1 - (x1 - x0) * 0.14
                ax.annotate(short, xy, xytext=(-6 if edge else 6, 5), textcoords="offset points", ha="right" if edge else "left", fontsize=7.4 if not air else 6.4,
                            fontweight="bold" if not air else "normal", color=b.ink, zorder=15, path_effects=halo(w=3))
        pg.site_pin(ax, c.site.xy, c.site.name, dx=-10, dy=-16, ha="right")
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        # distance table
        a = pg.panel((0.69, 0.07, 0.28, 0.835), "Distance from site")
        rows = sorted(((miles_between(c.site.xy, xy), n) for n, xy in dests), key=lambda t: t[0])
        y = 0.86
        for mi, n in rows:
            a.text(0.06, y, n.replace("George Bush Intercontinental Airport", "George Bush Intercontinental (IAH)").replace("William P. Hobby Airport", "William P. Hobby Airport"),
                   fontsize=7.4, color=b.ink, va="center", wrap=True, transform=a.transAxes)
            a.text(0.94, y, f"{mi:.0f} mi", fontsize=9, fontweight="bold", color=b.primary, ha="right", va="center", family=b.heading, transform=a.transAxes)
            a.plot([0.05, 0.95], [y - 0.034] * 2, color="#D8DDE3", lw=0.5, transform=a.transAxes)
            y -= 0.066
        a.text(0.06, 0.915, "STRAIGHT-LINE MILES FROM SITE", fontsize=5.8, color=b.muted, fontweight="bold", va="center", transform=a.transAxes)
        a.text(0.06, 0.225, "Drive times not shown: no routing service is connected.\nPlug one in via DriveTimeProvider.", fontsize=6.2, color=b.muted, va="top", transform=a.transAxes, linespacing=1.4)
        base.highway_legend(a, 0.06, 0.14)
        return pg
