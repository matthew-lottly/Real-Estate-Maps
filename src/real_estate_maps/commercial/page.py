"""Page: letter-landscape canvas (11 x 8.5 in = 3300 x 2550 px at 300 ppi) with the furniture every map shares."""
from __future__ import annotations

import textwrap
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon as MplPoly, Rectangle
from PIL import Image

from .brand import Brand
from .inputs import M_PER_MI, Site

W_IN, H_IN, DPI = 11.0, 8.5, 300


def halo(color="white", w=3.0):
    return [pe.withStroke(linewidth=w, foreground=color)]


class Page:
    HEADER_H = 0.085

    def __init__(self, brand: Brand, title: str, subtitle: str, source: str, stamp: str | None = None):
        self.brand, self.title, self.subtitle, self.source, self.stamp = brand, title, subtitle, source, stamp
        matplotlib.rcParams["font.family"] = brand.body
        matplotlib.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["text.parse_math"] = False
        matplotlib.rcParams["pdf.fonttype"] = 42
        self.fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=brand.paper)
        self.ax: plt.Axes | None = None
        self.issues: list[str] = []
        self._n = 0
        self._header()
        self._footer()

    # ---- furniture ----
    def _header(self) -> None:
        b, f = self.brand, self.fig
        a = f.add_axes((0, 1 - self.HEADER_H, 1, self.HEADER_H))
        a.set_axis_off()
        a.add_patch(Rectangle((0, 0), 1, 1, facecolor=b.primary, transform=a.transAxes))
        a.add_patch(Rectangle((0, 0), 1, 0.06, facecolor=b.accent, transform=a.transAxes))
        a.text(0.03, 0.60, self.title, color="white", fontsize=19, fontweight="bold", family=b.heading, va="center", transform=a.transAxes)
        a.text(0.03, 0.26, self.subtitle, color="#D9E2EA", fontsize=9.5, va="center", transform=a.transAxes)
        a.set_gid("header")
        if b.logo and Path(b.logo).exists():
            img = mpimg.imread(b.logo)
            h = 0.6
            la = f.add_axes((0.80, 1 - self.HEADER_H + self.HEADER_H * 0.2, 0.17, self.HEADER_H * h))
            la.imshow(img)
            la.set_axis_off()
        else:
            a.text(0.97, 0.5, b.name.upper(), color="white", fontsize=11, fontweight="bold", family=b.heading, ha="right", va="center", transform=a.transAxes)

    def _footer(self) -> None:
        f, b = self.fig, self.brand
        a = f.add_axes((0, 0, 1, 0.05))
        a.set_axis_off()
        a.plot([0.03, 0.97], [0.98, 0.98], color=b.muted, lw=0.5, alpha=0.5, transform=a.transAxes)
        a.text(0.03, 0.86, textwrap.fill(self.source, 215), fontsize=5.2, color=b.muted, va="top", linespacing=1.18, transform=a.transAxes)
        if self.stamp:
            a.text(0.97, 0.86, self.stamp, fontsize=6.0, color=b.alert, fontweight="bold", ha="right", va="top", transform=a.transAxes)
        a.set_gid("footer")

    # ---- map axes ----
    def map_axes(self, rect, center, width_m: float, facecolor: str | None = None) -> plt.Axes:
        """`rect` in figure fractions; the extent is `width_m` wide and as tall as the rect's aspect needs."""
        ax = self.fig.add_axes(rect)
        h_m = width_m * (rect[3] * H_IN) / (rect[2] * W_IN)
        ax.set_xlim(center[0] - width_m / 2, center[0] + width_m / 2)
        ax.set_ylim(center[1] - h_m / 2, center[1] + h_m / 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(facecolor or self.brand.water)
        for s in ax.spines.values():
            s.set_edgecolor(self.brand.primary)
            s.set_linewidth(1.0)
        self.ax = ax
        return ax

    @staticmethod
    def extent(ax) -> tuple[float, float, float, float]:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return x0, y0, x1, y1

    @contextmanager
    def layer(self, name: str, ax: plt.Axes | None = None):
        """Everything drawn inside gets a named group in the SVG, so the file opens as labelled layers."""
        ax = ax or self.ax
        before = set(ax.get_children())
        yield
        for a in set(ax.get_children()) - before:
            self._n += 1
            a.set_gid(f"{name}-{self._n}")

    def north_arrow(self, ax, corner="tr") -> None:
        x0, y0, x1, y1 = self.extent(ax)
        W, H = x1 - x0, y1 - y0
        cx, cy = (x1 - W * 0.04, y1 - H * 0.09) if corner == "tr" else (x0 + W * 0.04, y1 - H * 0.09)
        r = W * 0.024
        with self.layer("north-arrow", ax):
            ax.add_patch(Circle((cx, cy), r * 1.3, facecolor="white", edgecolor=self.brand.primary, lw=0.9, zorder=30))
            ax.add_patch(MplPoly([(cx, cy + r), (cx - r * .5, cy - r * .7), (cx, cy - r * .3)], closed=True, fc=self.brand.primary, ec=self.brand.primary, zorder=31))
            ax.add_patch(MplPoly([(cx, cy + r), (cx + r * .5, cy - r * .7), (cx, cy - r * .3)], closed=True, fc="white", ec=self.brand.primary, lw=0.8, zorder=31))
            ax.text(cx, cy + r * 1.45, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", color=self.brand.primary, zorder=31, path_effects=halo())

    def scale_bar(self, ax, unit="mi", corner="bl") -> None:
        """Pick a round length near 1/5 of the map width. unit: 'mi' or 'ft'."""
        x0, y0, x1, y1 = self.extent(ax)
        W, H = x1 - x0, y1 - y0
        unit_m = M_PER_MI if unit == "mi" else 0.3048
        target = W * 0.2 / unit_m
        totals = {"mi": [(0.5, 2), (1, 2), (2, 4), (3, 3), (5, 5), (10, 5), (20, 4), (25, 5), (50, 5)], "ft": [(100, 2), (200, 4), (500, 5), (1000, 5), (2000, 4)]}[unit]
        total, n = max((t for t in totals if t[0] <= target), default=totals[0])
        seg_len = total / n
        L = seg_len * unit_m * n
        bx, by, h = x0 + W * 0.035, y0 + H * 0.06, H * 0.011
        with self.layer("scale-bar", ax):
            ax.add_patch(Rectangle((bx - W * 0.012, by - H * 0.03), L + W * 0.07, H * 0.09, fc="white", alpha=0.85, ec="none", zorder=29))
            for i in range(n):
                ax.add_patch(Rectangle((bx + i * seg_len * unit_m, by), seg_len * unit_m, h, fc=self.brand.primary if i % 2 == 0 else "white", ec=self.brand.primary, lw=0.6, zorder=30))
            for i in range(n + 1):
                ax.text(bx + i * seg_len * unit_m, by + h * 1.6, f"{i * seg_len:g}", fontsize=6, ha="center", va="bottom", color=self.brand.ink, zorder=31)
            ax.text(bx + L + W * 0.008, by + h / 2, "miles" if unit == "mi" else "feet", fontsize=6, va="center", color=self.brand.ink, zorder=31)

    def in_furniture(self, ax, x: float, y: float) -> bool:
        """True when a point sits under the north arrow or scale bar (labels should be skipped there)."""
        x0, y0, x1, y1 = self.extent(ax)
        W, H = x1 - x0, y1 - y0
        return (x > x1 - 0.14 * W and y > y1 - 0.20 * H) or (x < x0 + 0.27 * W and y < y0 + 0.19 * H)

    def site_pin(self, ax, xy, label: str | None = None, size=11, dx=9, dy=9, ha="left") -> None:
        with self.layer("site-marker", ax):
            ax.plot(*xy, marker="*", ms=size * 1.7, mfc=self.brand.alert, mec="white", mew=1.2, zorder=40)
            if label:
                ax.annotate(label, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=7.5, fontweight="bold", family=self.brand.heading, color=self.brand.ink,
                            zorder=41, path_effects=halo(w=3.2))

    def badge(self, ax, xy, text: str, color: str | None = None, r_frac: float = 0.0105, fs: float = 6.5, zorder=35):
        x0, y0, x1, y1 = self.extent(ax)
        r = (x1 - x0) * r_frac
        ax.add_patch(Circle(xy, r, fc=color or self.brand.primary, ec="white", lw=1.0, zorder=zorder))
        ax.text(*xy, text, ha="center", va="center", color="white", fontsize=fs, fontweight="bold", zorder=zorder + 1)

    def panel(self, rect, title: str | None = None):
        a = self.fig.add_axes(rect)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_axis_off()
        a.add_patch(Rectangle((0, 0), 1, 1, fc="#F7F8FA", ec=self.brand.primary, lw=0.7, transform=a.transAxes))
        if title:
            hh = 0.30 / (rect[3] * H_IN)                      # header band is 0.30 in tall whatever the panel height
            a.add_patch(Rectangle((0, 1 - hh), 1, hh, fc=self.brand.primary, ec="none", transform=a.transAxes))
            a.text(0.04, 1 - hh / 2, title.upper(), color="white", fontsize=7.2, fontweight="bold", family=self.brand.heading, va="center", transform=a.transAxes)
        return a

    @staticmethod
    def dot(ax, x: float, y: float, text: str, color: str, ms: float = 9.5, fs: float = 6.0) -> None:
        """Round numbered dot in axes coordinates (a Circle patch would stretch to an ellipse on a non-square panel)."""
        ax.plot([x], [y], marker="o", ms=ms, color=color, mec="none", ls="none", transform=ax.transAxes, clip_on=False)
        ax.text(x, y, text, fontsize=fs, color="white", fontweight="bold", ha="center", va="center", transform=ax.transAxes)

    # ---- output ----
    def export(self, outdir: Path, stem: str) -> dict[str, Path]:
        """PNG at 300 ppi (3300 x 2550), PDF at page size, and an SVG with named layer groups (opens in Illustrator, Affinity, Inkscape)."""
        for a in self.fig.axes:
            a.set_xlabel("")
            a.set_ylabel("")
        from .qa import audit
        self.issues = audit(self.fig)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {k: outdir / f"{stem}.{k}" for k in ("png", "pdf", "svg")}
        self.fig.savefig(paths["png"], dpi=DPI, facecolor=self.brand.paper)
        im = Image.open(paths["png"])
        assert im.size == (3300, 2550), f"unexpected PNG size {im.size}"
        im.save(paths["png"], dpi=(DPI, DPI), optimize=True)
        self.fig.savefig(paths["pdf"], facecolor=self.brand.paper)
        self.fig.savefig(paths["svg"], facecolor=self.brand.paper)
        plt.close(self.fig)
        return paths
