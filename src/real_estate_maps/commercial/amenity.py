from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle

from .base import BaseMap, CommercialMap, miles_between
from .inputs import M_PER_MI
from .page import Page, halo

SRC = ("Water: Natural Earth (public domain); ZIP outlines: Census 2010 ZCTAs. Street-level road data is not available in this build. Amenity list supplied by the broker (sample list in this demo: three retail centers and a park use approximate coordinates; "
       "restaurants and employers are labelled samples). Logos: supplied files where present, otherwise a monogram badge. Verify names and positions against the supplied list before publishing.")


class AmenityMap(CommercialMap):
    key = "amenities"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        pg = Page(b, "Area Amenities", f"{c.site.name}  ·  retail, dining and employers within about 6 miles", SRC, c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), c.site.xy, 20_000)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        base.outlines(ax)
        x0, y0, x1, y1 = pg.extent(ax)
        sx, sy = c.site.xy
        with pg.layer("distance-rings", ax):
            for mi in (2, 4):
                ax.add_patch(Circle((sx, sy), mi * M_PER_MI, fc="none", ec=b.primary, lw=0.7, ls=(0, (5, 4)), alpha=0.6, zorder=5))
                ax.text(sx, sy + mi * M_PER_MI, f"{mi} mi", fontsize=6.2, color=b.primary, ha="center", va="bottom", fontweight="bold", zorder=6, path_effects=halo())
        inside = [a for a in c.amenities if x0 < a.xy[0] < x1 and y0 < a.xy[1] < y1]
        inside.sort(key=lambda a: -a.xy[1])
        r = (x1 - x0) * 0.0135
        with pg.layer("amenity-markers", ax):
            for i, a in enumerate(inside, 1):
                col = b.categories.get(a.category, b.muted)
                ax.add_patch(Circle(a.xy, r, fc="white", ec=col, lw=1.8, zorder=20))
                if a.logo:
                    ab = AnnotationBbox(OffsetImage(mpimg.imread(a.logo), zoom=0.12), a.xy, frameon=False, zorder=21)
                    ax.add_artist(ab)
                else:
                    ax.text(*a.xy, "".join(w[0] for w in a.name.split()[:2]).upper(), fontsize=6.4, fontweight="bold", color=col, ha="center", va="center", zorder=21)
                ax.text(a.xy[0], a.xy[1] - r * 1.35, str(i), fontsize=5.5, fontweight="bold", color="white", ha="center", va="center", zorder=22,
                        bbox=dict(boxstyle="circle,pad=0.18", fc=b.primary, ec="white", lw=0.6))
        pg.site_pin(ax, c.site.xy, c.site.name)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        pan = pg.panel((0.69, 0.07, 0.28, 0.835), "Nearby")
        y = 0.9
        for i, a in enumerate(inside, 1):
            col = b.categories.get(a.category, b.muted)
            pan.add_patch(Circle((0.075, y), 0.022, fc=col, ec="none", transform=pan.transAxes))
            pan.text(0.075, y, str(i), fontsize=6, color="white", fontweight="bold", ha="center", va="center", transform=pan.transAxes)
            pan.text(0.14, y + 0.006, a.name, fontsize=7.4, color=b.ink, fontweight="bold", va="center", transform=pan.transAxes)
            pan.text(0.14, y - 0.017, a.category.title(), fontsize=5.8, color=b.muted, va="center", transform=pan.transAxes)
            pan.text(0.94, y, f"{miles_between(c.site.xy, a.xy):.1f} mi", fontsize=7.6, color=b.primary, fontweight="bold", ha="right", va="center", transform=pan.transAxes)
            y -= 0.052
        y -= 0.02
        for k, col in b.categories.items():
            if any(a.category == k for a in inside):
                pan.add_patch(Circle((0.07, y), 0.014, fc=col, ec="none", transform=pan.transAxes))
                pan.text(0.11, y, k.title(), fontsize=6.4, color=b.ink, va="center", transform=pan.transAxes)
                y -= 0.032
        return pg
