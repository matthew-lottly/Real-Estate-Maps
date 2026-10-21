from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import box

from ..classification import ClassifierFactory
from ..config import PLACES
from ..metrics import MetricRegistry
from .base import BaseMap, CommercialMap
from .inputs import to_xy
from .page import Page, halo

SRC = ("Median sale price: Redfin Data Center ZIP-code tracker, all residential, rolling 3 months ending {end}. Areas are ZIP codes (Census 2010 ZCTAs), the finest level this data supports — not neighborhood "
       "boundaries. ZIPs with fewer than 10 sales are hatched. Streets: Overture Maps.")


class NeighborhoodMap(CommercialMap):
    key = "neighborhood"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        m = MetricRegistry.get("price")
        end = c.dataset.window_label().split("–")[-1].strip()
        pg = Page(b, "Neighborhood Market Map", f"{c.site.name}  ·  median home sale price by ZIP, 3 months ending {end}", SRC.format(end=end), c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), c.site.xy, 26_000)
        z = c.zips.copy()
        z["ok"] = z[m.column].notna() & (z["homes_sold"].fillna(0) >= m.min_sales)
        scheme = ClassifierFactory.for_metric(m).scheme(z.loc[z["ok"], m.column].to_numpy())
        z["cls"] = np.where(z["ok"], scheme.assign(z[m.column].to_numpy()), -1)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        with pg.layer("zip-choropleth", ax):
            for i, col in enumerate(scheme.colors):
                z[z["cls"] == i].plot(ax=ax, fc=col, ec="white", lw=0.9, zorder=2)
            nd = z[z["cls"] < 0]
            if len(nd):
                nd.plot(ax=ax, fc="#E4E6E9", ec="#B3BAC3", hatch="////", lw=0.3, zorder=2)
                nd.boundary.plot(ax=ax, color="white", lw=0.9, zorder=2.1)
        base.streets(ax, label=False, light=True)
        x0, y0, x1, y1 = pg.extent(ax)
        view = box(x0, y0, x1, y1).buffer(-0.04 * (x1 - x0))
        rows = []
        with pg.layer("zip-labels", ax):
            for _, r in z.iterrows():
                p = r.geometry.intersection(view)
                if p.is_empty or p.area < r.geometry.area * 0.25:
                    continue
                pt = p.representative_point()
                if pg.in_furniture(ax, pt.x, pt.y):
                    continue
                ax.text(pt.x, pt.y + (y1 - y0) * 0.008, r["zip"], fontsize=7, fontweight="bold", color=b.ink, ha="center", va="bottom", zorder=15, path_effects=halo("white", 2.6))
                if r["ok"]:
                    ax.text(pt.x, pt.y - (y1 - y0) * 0.004, m.short(r[m.column]), fontsize=6.4, color=b.ink, ha="center", va="top", zorder=15, path_effects=halo("white", 2.4))
                    rows.append((r["zip"], r[m.column], r["cls"]))
        with pg.layer("place-labels", ax):
            for n, lo, la, rk in PLACES:
                px, py = to_xy(lo, la)
                if x0 < px < x1 and y0 < py < y1:
                    ax.plot(px, py, "s", ms=3.5, color=b.ink, zorder=16)
                    edge = px > x1 - (x1 - x0) * 0.12
                    ax.annotate(n, (px, py), xytext=(-5 if edge else 5, -9), ha="right" if edge else "left", textcoords="offset points", fontsize=6.2, style="italic", color=b.muted, zorder=16, path_effects=halo())
        pg.site_pin(ax, c.site.xy, c.site.name)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        pan = pg.panel((0.69, 0.07, 0.28, 0.835), "Median sale price")
        y = 0.87
        for col, lab in zip(reversed(scheme.colors), reversed(scheme.labels)):
            pan.add_patch(Rectangle((0.06, y - 0.017), 0.09, 0.034, fc=col, ec="white", transform=pan.transAxes))
            pan.text(0.19, y, lab, fontsize=7.2, color=b.ink, va="center", transform=pan.transAxes)
            y -= 0.048
        pan.add_patch(Rectangle((0.06, y - 0.017), 0.09, 0.034, fc="#E4E6E9", ec="#B3BAC3", hatch="////", transform=pan.transAxes))
        pan.text(0.19, y, "Under 10 sales / no data", fontsize=7.2, color=b.ink, va="center", transform=pan.transAxes)
        y -= 0.085
        pan.text(0.06, y, "ZIPS ON THIS MAP", fontsize=6.2, fontweight="bold", color=b.primary, va="center", transform=pan.transAxes)
        y -= 0.04
        for zc, v, k in sorted(rows, key=lambda t: -t[1])[:10]:
            pan.text(0.06, y, zc, fontsize=7.2, color=b.ink, fontweight="bold", va="center", transform=pan.transAxes)
            pan.text(0.94, y, m.fmt(v), fontsize=7.2, color=b.ink, ha="right", va="center", transform=pan.transAxes)
            y -= 0.038
        return pg
