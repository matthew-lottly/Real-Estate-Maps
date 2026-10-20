from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from shapely.geometry import Point

from ..classification import ClassifierFactory
from ..metrics import MetricRegistry
from .base import BaseMap, CommercialMap
from .page import Page, halo

SRC = ("Redfin Data Center ZIP-code tracker, all residential, rolling 3-month windows ending {end}. Fill is the change in median sale price vs the same window a year earlier; ZIPs with fewer than 10 sales are hatched. "
       "Lines show median sale price indexed to 100 five years earlier and smoothed with a 4-window moving average; the metro line is a sales-weighted average of ZIP medians. ZIP areas are Census 2010 ZCTAs. Small ZIPs are volatile.")


class MarketTrendsMap(CommercialMap):
    """Where prices are moving around the site, with the nearest ZIPs' five-year price paths against the metro."""

    key = "market_trends"

    def build(self) -> Page:
        c, b = self.c, self.c.brand
        m = MetricRegistry.get("yoy")
        end = c.dataset.window_label().split("–")[-1].strip()
        pg = Page(b, "Market Momentum", f"{c.site.name}  ·  price change vs a year earlier, 3 months ending {end}", SRC.format(end=end), c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.56, 0.835), c.site.xy, 30_000)
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
            nd.plot(ax=ax, fc="#E4E6E9", ec="#B3BAC3", hatch="////", lw=0.3, zorder=2)
            nd.boundary.plot(ax=ax, color="white", lw=0.9, zorder=2.1)
        sp = Point(c.site.xy)
        near = z[z["ok"]].assign(d=lambda d: d.geometry.distance(sp)).sort_values("d").head(4)
        x0, y0, x1, y1 = pg.extent(ax)
        lines = ["#14304A", "#1F8A8A", "#C9A24B", "#D1493B"]
        with pg.layer("zip-labels", ax):
            for (_, r), col in zip(near.iterrows(), lines):
                pt = r.geometry.representative_point()
                ax.text(pt.x, pt.y, r["zip"], fontsize=8, fontweight="bold", color="white", ha="center", va="center", zorder=15,
                        bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="white", lw=0.8))
                ax.text(pt.x, pt.y - (y1 - y0) * 0.028, m.fmt(r[m.column]), fontsize=7, fontweight="bold", color=b.ink, ha="center", va="top", zorder=15, path_effects=halo("white", 2.6))
        pg.site_pin(ax, c.site.xy, c.site.name, dx=10, dy=16)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        # right side: legend + line chart
        pan = pg.panel((0.61, 0.07, 0.36, 0.835), "Price change vs year earlier")
        yy = 0.88
        for col, lab in zip(reversed(scheme.colors), reversed(scheme.labels)):
            pan.add_patch(Rectangle((0.05, yy - 0.012), 0.06, 0.024, fc=col, ec="white", transform=pan.transAxes))
            pan.text(0.13, yy, lab, fontsize=6.6, color=b.ink, va="center", transform=pan.transAxes)
            yy -= 0.034
        pan.text(0.05, 0.60, "FIVE-YEAR PRICE PATH (INDEX, START = 100, 4-MONTH AVG)", fontsize=6.2, fontweight="bold", color=b.primary, va="center", transform=pan.transAxes)
        ch = pg.fig.add_axes((0.64, 0.12, 0.30, 0.37))
        mk = c.dataset.market
        mk = mk[mk["median_sale_price"].notna()]
        last = pd.Timestamp(c.dataset.latest)
        mk = mk[pd.to_datetime(mk["period_end"]) >= last - pd.DateOffset(years=5)]
        def idx(df):
            s = df.sort_values("period_end").set_index(pd.to_datetime(df.sort_values("period_end")["period_end"]))["median_sale_price"]
            return s / s.iloc[0] * 100
        metro = mk.groupby("period_end").apply(lambda g: (g["median_sale_price"] * g["homes_sold"]).sum() / g["homes_sold"].sum(), include_groups=False)
        metro.index = pd.to_datetime(metro.index)
        ch.plot(metro.index, (metro / metro.iloc[0] * 100).rolling(4, min_periods=1).mean(), color="#98A1AB", lw=2.2, ls="--")
        ch.annotate("Metro", (metro.index[-1], (metro / metro.iloc[0] * 100).rolling(4, min_periods=1).mean().iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=b.muted, va="center")
        for (_, r), col in zip(near.iterrows(), lines):
            g = mk[mk["zip"] == r["zip"]]
            if len(g) > 12:
                s = idx(g).rolling(4, min_periods=1).mean()
                ch.plot(s.index, s.values, color=col, lw=1.8)
                ch.annotate(r["zip"], (s.index[-1], s.values[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=col, fontweight="bold", va="center")
        ch.set_xlim(right=last + pd.Timedelta(days=170))
        ch.tick_params(labelsize=6.2, colors=b.muted, length=2)
        ch.grid(axis="y", color="#E1E5EA", lw=0.6)
        for s_ in ("top", "right"):
            ch.spines[s_].set_visible(False)
        ch.set_facecolor("#F7F8FA")
        return pg
