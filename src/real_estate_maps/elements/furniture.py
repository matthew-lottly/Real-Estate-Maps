from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from ..composer import MapElement, RenderContext
from .layers import halo


class Graticule(MapElement):
    """Faint lat/lon grid with coordinate labels along the map edge."""

    name = "graticule"

    def draw(self, c: RenderContext) -> None:
        minx, miny, maxx, maxy = c.extent
        lons, lats = np.arange(-96.2, -94.5, 0.2), np.arange(29.2, 30.5, 0.2)
        for lon in lons:
            pts = np.array([c.to_xy(lon, la) for la in np.linspace(29.0, 30.6, 160)])
            c.ax.plot(pts[:, 0], pts[:, 1], color="white", lw=0.8, alpha=0.7, zorder=1.5)
            inside = pts[(pts[:, 1] > miny) & (pts[:, 1] < maxy) & (pts[:, 0] > minx) & (pts[:, 0] < maxx)]
            if len(inside):
                p = inside[np.argmin(inside[:, 1])]
                c.ax.text(p[0], miny + (maxy - miny) * 0.006, f"{abs(lon):.1f}°W", fontsize=7, color=c.theme.muted, ha="center", va="bottom", zorder=10,
                          path_effects=halo(c.theme.water, 2.5))
        for la in lats:
            pts = np.array([c.to_xy(lo, la) for lo in np.linspace(-96.4, -94.4, 160)])
            c.ax.plot(pts[:, 0], pts[:, 1], color="white", lw=0.8, alpha=0.7, zorder=1.5)
            inside = pts[(pts[:, 1] > miny) & (pts[:, 1] < maxy) & (pts[:, 0] > minx) & (pts[:, 0] < maxx)]
            if len(inside):
                p = inside[np.argmin(inside[:, 0])]
                c.ax.text(minx + (maxx - minx) * 0.006, p[1], f"{la:.1f}°N", fontsize=7, color=c.theme.muted, ha="left", va="center", zorder=10,
                          path_effects=halo(c.theme.water, 2.5))


class NorthArrow(MapElement):
    name = "north_arrow"

    def draw(self, c: RenderContext) -> None:
        minx, miny, maxx, maxy = c.extent
        w = maxx - minx
        cx, cy, r = maxx - w * 0.055, maxy - (maxy - miny) * 0.085, w * 0.032
        c.ax.add_patch(Circle((cx, cy), r * 1.25, facecolor="white", edgecolor=c.theme.ink, linewidth=1.1, alpha=0.95, zorder=11))
        c.ax.add_patch(Polygon([(cx, cy + r), (cx - r * 0.55, cy - r * 0.75), (cx, cy - r * 0.3)], closed=True, facecolor=c.theme.ink, edgecolor=c.theme.ink, zorder=12))
        c.ax.add_patch(Polygon([(cx, cy + r), (cx + r * 0.55, cy - r * 0.75), (cx, cy - r * 0.3)], closed=True, facecolor="white", edgecolor=c.theme.ink, linewidth=1, zorder=12))
        c.ax.text(cx, cy + r * 1.55, "N", ha="center", va="bottom", fontsize=12, fontweight="bold", color=c.theme.ink, zorder=12, path_effects=halo(w=3))


class ScaleBar(MapElement):
    """Kilometre bar and mile bar, drawn in map units (the projection is in metres)."""

    name = "scale_bar"

    def _bar(self, c: RenderContext, x0: float, y0: float, unit_m: float, n: int, seg: float, label: str, above: bool = True) -> None:
        h = (c.extent[3] - c.extent[1]) * 0.0075
        for i in range(n):
            c.ax.add_patch(Rectangle((x0 + i * seg * unit_m, y0), seg * unit_m, h, facecolor=c.theme.ink if i % 2 == 0 else "white", edgecolor=c.theme.ink, linewidth=0.9, zorder=12))
        for i in range(n + 1):
            c.ax.text(x0 + i * seg * unit_m, y0 + h * 1.8 if above else y0 - h * 0.8, f"{i * seg:g}", fontsize=7.5, ha="center", va="bottom" if above else "top", color=c.theme.ink, zorder=12, path_effects=halo(w=2.5))
        c.ax.text(x0 + n * seg * unit_m + (c.extent[2] - c.extent[0]) * 0.006, y0 + h / 2, label, fontsize=7.5, ha="left", va="center", color=c.theme.ink, zorder=12, path_effects=halo(w=2.5))

    def draw(self, c: RenderContext) -> None:
        minx, miny, maxx, maxy = c.extent
        H, W = maxy - miny, maxx - minx
        x0, y0 = minx + W * 0.07, miny + H * 0.075
        c.ax.add_patch(Rectangle((x0 - W * 0.014, y0 - H * 0.032), W * 0.225, H * 0.115, facecolor="white", alpha=0.78, edgecolor="none", zorder=11))
        self._bar(c, x0, y0 + H * 0.03, 1000.0, 4, 5, "km")
        self._bar(c, x0, y0, 1609.344, 4, 3, "mi", above=False)
