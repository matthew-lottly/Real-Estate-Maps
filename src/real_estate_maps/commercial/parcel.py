from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Polygon as MplPoly
from shapely.geometry import Point

from .base import CommercialMap
from .inputs import M_PER_FT
from .page import Page, halo

SRC = ("Parcel boundary: supplied by the broker (this demo uses a synthetic rectangle, not a real tax lot). Dimensions and acreage are computed from the boundary geometry. "
       "Aerial imagery: licensed imagery is not bundled; drop aerial.png + aerial.json into the inputs folder and it is placed automatically.")


class AerialParcelMap(CommercialMap):
    key = "aerial_parcel"

    def build(self) -> Page:
        c, b, s = self.c, self.c.brand, self.c.site
        pg = Page(b, "Aerial Parcel Map", f"{s.name}  ·  {s.address}", SRC, c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), s.parcel.centroid.coords[0], 360, facecolor="#2E3B38")
        ext = pg.extent(ax)
        with pg.layer("imagery", ax):
            if not c.imagery.draw(ax, ext):
                x0, y0, x1, y1 = ext
                for k in np.linspace(x0, x1, 21):
                    ax.plot([k, k], [y0, y1], color="#3D4D49", lw=0.5, zorder=1)
                for k in np.linspace(y0, y1, 21):
                    ax.plot([x0, x1], [k, k], color="#3D4D49", lw=0.5, zorder=1)
                ax.text((x0 + x1) / 2, y1 - (y1 - y0) * 0.06, "AERIAL IMAGERY PLACEHOLDER — licensed imagery not installed", color="#C9D3D0", fontsize=7.5, ha="center",
                        fontweight="bold", zorder=2)
        coords = list(s.parcel.exterior.coords)
        with pg.layer("parcel-boundary", ax):
            ax.add_patch(MplPoly(coords, closed=True, fc=b.accent, alpha=0.16, ec="none", zorder=10))
            ax.add_patch(MplPoly(coords, closed=True, fc="none", ec="black", lw=4.2, alpha=0.35, zorder=11))
            ax.add_patch(MplPoly(coords, closed=True, fc="none", ec=b.accent, lw=2.4, zorder=12))
        with pg.layer("dimension-labels", ax):
            for (xa, ya), (xb, yb) in zip(coords[:-1], coords[1:]):
                ln = np.hypot(xb - xa, yb - ya) / M_PER_FT
                ang = np.degrees(np.arctan2(yb - ya, xb - xa))
                ang = ang + 180 if ang > 90 or ang < -90 else ang
                cx, cy = (xa + xb) / 2, (ya + yb) / 2
                ax.text(cx, cy, f"{ln:,.0f} ft", rotation=ang, rotation_mode="anchor", fontsize=7, fontweight="bold", color="white", ha="center", va="bottom", zorder=15,
                        path_effects=halo("#101615", 3), bbox=None)
            cx, cy = s.parcel.centroid.coords[0]
            ax.text(cx, cy, f"{s.acres:.2f} AC", fontsize=13, fontweight="bold", family=b.heading, color="white", ha="center", va="center", zorder=15, path_effects=halo("#101615", 3.5))
        # numbered callouts around the parcel
        x0, y0, x1, y1 = ext
        W, H = x1 - x0, y1 - y0
        anchors = [(0.13, 0.80), (0.87, 0.80), (0.13, 0.20), (0.87, 0.20)]
        cxm, cym = s.parcel.centroid.coords[0]
        with pg.layer("callouts", ax):
            for i, (txt, (fx, fy)) in enumerate(zip(s.callouts, anchors), 1):
                tx, ty = x0 + W * fx, y0 + H * fy
                px, py = s.parcel.exterior.interpolate(s.parcel.exterior.project(Point(tx, ty))).coords[0]
                ax.plot([tx, px], [ty, py], color="white", lw=1.0, zorder=16)
                ax.plot(px, py, "o", ms=3, color="white", zorder=17)
                pg.badge(ax, (tx, ty), str(i), color=b.primary, r_frac=0.016, fs=8, zorder=18)
        pg.north_arrow(ax)
        pg.scale_bar(ax, unit="ft")
        pan = pg.panel((0.69, 0.07, 0.28, 0.835), "Site highlights")
        pan.text(0.06, 0.90, s.name, fontsize=11, fontweight="bold", family=b.heading, color=b.primary, va="top", transform=pan.transAxes)
        pan.text(0.06, 0.855, s.address, fontsize=6.6, color=b.muted, va="top", wrap=True, transform=pan.transAxes)
        pan.text(0.06, 0.78, f"{s.acres:.2f}", fontsize=24, fontweight="bold", family=b.heading, color=b.accent, va="top", transform=pan.transAxes)
        pan.text(0.06, 0.715, "ACRES (from boundary geometry)", fontsize=5.8, color=b.muted, fontweight="bold", va="top", transform=pan.transAxes)
        y = 0.64
        for i, t in enumerate(s.callouts, 1):
            pan.add_patch(Circle((0.085, y), 0.026, fc=b.primary, ec="none", transform=pan.transAxes))
            pan.text(0.085, y, str(i), fontsize=7, color="white", fontweight="bold", ha="center", va="center", transform=pan.transAxes)
            pan.text(0.16, y, t, fontsize=7.4, color=b.ink, va="center", transform=pan.transAxes)
            y -= 0.075
        return pg
