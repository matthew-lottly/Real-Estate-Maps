from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Point

from .base import BaseMap, CommercialMap
from .inputs import M_PER_MI
from .page import Page, halo

SRC = ("Housing-market figures: Redfin Data Center ZIP-code tracker, all residential, 3 months ending {end}. ZIP values are apportioned to each ring by area overlap (assumes sales are spread evenly across a ZIP), "
       "so they are estimates; the price is a sales-weighted average of ZIP medians, not a true ring median. ZIP areas are Census 2010 ZCTAs; streets: Overture Maps. "
       "Population, household and income figures appear only when a demographic extract is supplied (inputs/demographics.csv); none is bundled.")


class RingStats:
    """Area-apportioned housing stats for a ring around the site."""

    def __init__(self, zips, center):
        self.z, self.center = zips[zips["median_sale_price"].notna() & zips["homes_sold"].notna()].copy(), center

    def of(self, miles: float) -> dict:
        ring = Point(self.center).buffer(miles * M_PER_MI, 128)
        z = self.z
        frac = z.geometry.intersection(ring).area / z.geometry.area
        w = frac * z["homes_sold"]
        hit = frac > 0.005
        return {"zips": int(hit.sum()), "sold": float(w.sum()), "price": float((z["median_sale_price"] * w).sum() / w.sum()) if w.sum() else float("nan")}


class DemographicRingsMap(CommercialMap):
    key = "demographic_rings"
    RINGS = (1, 3, 5)

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        end = c.dataset.window_label().split("–")[-1].strip()
        pg = Page(b, "Demographic Rings", f"{c.site.name}  ·  1, 3 and 5 mile rings", SRC.format(end=end), c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.60, 0.835), c.site.xy, 21_000)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        sx, sy = c.site.xy
        with pg.layer("zip-outlines", ax):
            c.zips.boundary.plot(ax=ax, color="#BDB6A8", lw=0.5, zorder=2)
        tints = {5: b.secondary, 3: b.secondary, 1: b.secondary}
        alphas = {5: 0.10, 3: 0.14, 1: 0.20}
        base.streets(ax, label=False)
        with pg.layer("rings", ax):
            for mi in sorted(self.RINGS, reverse=True):
                ax.add_patch(Circle((sx, sy), mi * M_PER_MI, fc=tints[mi], alpha=alphas[mi], ec="none", zorder=4))
                ax.add_patch(Circle((sx, sy), mi * M_PER_MI, fc="none", ec=b.primary, lw=1.6, zorder=6))
                ang = np.radians(48)
                px, py = sx + mi * M_PER_MI * np.cos(ang), sy + mi * M_PER_MI * np.sin(ang)
                ax.text(px, py, f"{mi} MI", fontsize=7.5, fontweight="bold", color="white", ha="center", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc=b.primary, ec="white", lw=0.8))
        pg.site_pin(ax, c.site.xy, c.site.name, dx=8, dy=-16)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        rs = RingStats(c.zips, c.site.xy)
        metro = float((rs.z["median_sale_price"] * rs.z["homes_sold"]).sum() / rs.z["homes_sold"].sum())
        for i, mi in enumerate(self.RINGS):
            st = rs.of(mi)
            demo = c.demographics.get(mi)
            bottom = 0.105 + (2 - i) * 0.2675
            a = pg.panel((0.65, bottom, 0.32, 0.25))
            a.add_patch(Rectangle((0, 0), 0.025, 1, fc=b.primary, ec="none", transform=a.transAxes))
            a.text(0.07, 0.90, f"{mi} MILE", fontsize=7.4, fontweight="bold", family=b.heading, color=b.primary, va="center", transform=a.transAxes)
            a.text(0.93, 0.90, f"{st['zips']} ZIP areas touched", fontsize=5.8, color=b.muted, ha="right", va="center", transform=a.transAxes)
            cells = [("Homes sold (3 mo, est.)", f"{st['sold']:,.0f}"), ("Weighted median price", f"${st['price']:,.0f}"),
                     ("Est. sales volume", f"${st['sold'] * st['price'] / 1e6:,.0f}M"), ("Price vs metro avg", f"{(st['price'] / metro - 1) * 100:+.0f}%")]
            cells[2:] = [(k.replace("_", " ").title(), v) for k, v in list(demo.items())[:2]] or cells[2:]
            for j, (lab, val) in enumerate(cells[:4]):
                cx, cy = 0.07 + (j % 2) * 0.46, 0.60 - (j // 2) * 0.36
                a.text(cx, cy, val, fontsize=17, fontweight="bold", family=b.heading, color=b.ink, va="center", transform=a.transAxes)
                a.text(cx, cy - 0.15, lab.upper(), fontsize=5, fontweight="bold", color=b.muted, va="center", transform=a.transAxes)
        if not c.demographics.available:
            pg.fig.text(0.65, 0.075, "Population / income not shown — no demographic extract supplied.", fontsize=5.6, color=b.muted, va="bottom")
        return pg
