from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle

from .base import BaseMap, CommercialMap, miles_between
from .inputs import M_PER_MI
from .page import Page, halo

SRC = ("Amenities: Overture Maps Foundation places (release 2026-09-23.1; open businesses, confidence 0.8+), nearest per category; names and positions as published there, so verify against the broker's list before print. "
       "Water: Natural Earth. Logos: supplied files where present, otherwise a monogram badge. Streets: Overture Maps (same release).")


class AmenityMap(CommercialMap):
    key = "amenities"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        pg = Page(b, "Area Amenities", f"{c.site.name}  ·  retail, dining and employers within about 2 miles", SRC, c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), c.site.xy, 7_500)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        base.streets(ax)
        x0, y0, x1, y1 = pg.extent(ax)
        sx, sy = c.site.xy
        with pg.layer("distance-rings", ax):
            for mi in (1, 2):
                ax.add_patch(Circle((sx, sy), mi * M_PER_MI, fc="none", ec=b.primary, lw=0.7, ls=(0, (5, 4)), alpha=0.6, zorder=5))
                ax.text(sx, sy + mi * M_PER_MI, f"{mi} mi", fontsize=6.2, color=b.primary, ha="center", va="bottom", fontweight="bold", zorder=6, path_effects=halo())
        inside = [a for a in c.amenities if x0 < a.xy[0] < x1 and y0 < a.xy[1] < y1]
        inside.sort(key=lambda a: -a.xy[1])
        r = (x1 - x0) * 0.016
        pos = {id(a_): np.array(a_.xy, float) for a_ in inside}
        for _ in range(60):                                   # nudge overlapping badges apart (true position kept as a small dot)
            moved = False
            for i1, a1 in enumerate(inside):
                for a2 in inside[i1 + 1:]:
                    d = pos[id(a2)] - pos[id(a1)]
                    dist = float(np.hypot(*d))
                    if dist < 2.3 * r:
                        v = d / dist if dist > 1e-6 else np.array([1.0, 0.3])
                        push = (2.3 * r - dist) / 2
                        pos[id(a1)] -= v * push; pos[id(a2)] += v * push
                        moved = True
            if not moved:
                break
        with pg.layer("amenity-markers", ax):
            for i, a in enumerate(inside, 1):
                p = pos[id(a)]
                if np.hypot(*(p - np.array(a.xy))) > 0.4 * r:
                    ax.plot([a.xy[0], p[0]], [a.xy[1], p[1]], color=b.ink, lw=0.7, zorder=19)
                    ax.plot(*a.xy, "o", ms=2.5, color=b.ink, zorder=19)
                col = b.categories.get(a.category, b.muted)
                ax.add_patch(Circle(p, r, fc=col, ec="white", lw=1.4, zorder=20))
                if a.logo:
                    ax.add_artist(AnnotationBbox(OffsetImage(mpimg.imread(a.logo), zoom=0.12), p, frameon=False, zorder=21))
                else:
                    ax.text(*p, str(i), fontsize=6.8, fontweight="bold", color="white", ha="center", va="center", zorder=21)
        pg.site_pin(ax, c.site.xy, c.site.name)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        pan = pg.panel((0.69, 0.07, 0.28, 0.835), "Nearby")
        y = 0.9
        for i, a in enumerate(inside, 1):
            col = b.categories.get(a.category, b.muted)
            pg.dot(pan, 0.075, y, str(i), col, 10, 6)
            pan.text(0.14, y + 0.006, a.name, fontsize=7.4, color=b.ink, fontweight="bold", va="center", transform=pan.transAxes)
            pan.text(0.14, y - 0.017, a.category.title(), fontsize=5.8, color=b.muted, va="center", transform=pan.transAxes)
            pan.text(0.94, y, f"{miles_between(c.site.xy, a.xy):.1f} mi", fontsize=7.6, color=b.primary, fontweight="bold", ha="right", va="center", transform=pan.transAxes)
            y -= 0.046
        y -= 0.012
        for k, col in b.categories.items():
            if any(a.category == k for a in inside):
                pan.plot([0.07], [y], marker="o", ms=6, color=col, mec="none", ls="none", transform=pan.transAxes)
                pan.text(0.11, y, k.title(), fontsize=6.4, color=b.ink, va="center", transform=pan.transAxes)
                y -= 0.028
        return pg
