from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from ..palettes import PaletteLibrary
from .base import BaseMap, CommercialMap
from .inputs import M_PER_MI, to_xy
from .page import Page, halo

SRC = ("Screening score, not a site verdict. Built only from public inputs available here: Redfin ZIP housing market ({end}; used as proxies for demand, land cost and liquidity — not commercial rents, zoning, "
       "flood risk or parcel availability) and straight-line distance to generalized Natural Earth highways and airports. Each factor is converted to a 0–100 percentile across the metro's ZIPs, then weighted.")


@dataclass(frozen=True)
class Factor:
    key: str
    label: str
    higher_is_better: bool


@dataclass(frozen=True)
class Profile:
    """A use case = a weighted set of factors. Add a Profile to score a new asset type."""

    key: str
    title: str
    blurb: str
    weights: dict[str, float]


FACTORS = {
    "hwy": Factor("hwy", "Highway access (distance to Interstate/Beltway/US)", False),
    "air": Factor("air", "Airport access (IAH / Hobby)", False),
    "price_hi": Factor("price_hi", "Higher home prices (spending-power proxy)", True),
    "price_lo": Factor("price_lo", "Lower home prices (land-cost proxy)", False),
    "sales": Factor("sales", "Market activity (homes sold)", True),
    "yoy": Factor("yoy", "Price momentum (year over year)", True),
    "dom": Factor("dom", "Liquidity (shorter days on market)", False),
}
PROFILES = {
    "retail": Profile("retail", "Retail Site Suitability", "Neighborhood retail: spending power, activity and access", {"price_hi": .30, "sales": .25, "hwy": .20, "yoy": .15, "dom": .10}),
    "industrial": Profile("industrial", "Industrial / Logistics Suitability", "Light industrial and distribution: access first, cheaper land", {"hwy": .40, "air": .20, "price_lo": .25, "dom": .15}),
}
LABELS = ["Low", "Fair", "Moderate", "Good", "High"]


class SuitabilityModel:
    """Weighted overlay over ZIP polygons. Percentile-ranks each factor, then sums weight x score."""

    def __init__(self, ctx):
        z = ctx.zips.copy()
        z = z[z["median_sale_price"].notna() & (z["homes_sold"].fillna(0) >= 10)].copy()
        major = ctx.roads[(ctx.roads["level"] == "Interstate") | (ctx.roads["type"] == "Beltway") | ((ctx.roads["level"] == "Federal") & (ctx.roads["type"] == "Major Highway"))]
        hw = unary_union(list(major.geometry))
        ap = [to_xy(-95.3368, 29.9902), to_xy(-95.2789, 29.6454)]
        cen = z.geometry.centroid
        z["hwy"] = [hw.distance(p) / M_PER_MI for p in cen]
        z["air"] = [min(np.hypot(p.x - a[0], p.y - a[1]) for a in ap) / M_PER_MI for p in cen]
        self.z = z

    @staticmethod
    def _pct(s: pd.Series, higher_better: bool) -> pd.Series:
        r = s.rank(pct=True) * 100
        return r if higher_better else 100 - r + 100 / len(s)

    def score(self, prof: Profile) -> pd.DataFrame:
        z = self.z.copy()
        cols = {"hwy": z["hwy"], "air": z["air"], "price_hi": z["median_sale_price"], "price_lo": z["median_sale_price"], "sales": z["homes_sold"], "yoy": z["price_yoy"], "dom": z["median_dom"]}
        tot, wsum = pd.Series(0.0, index=z.index), pd.Series(0.0, index=z.index)
        for k, w in prof.weights.items():
            s = cols[k]
            p = self._pct(s.fillna(s.median()), FACTORS[k].higher_is_better)
            tot += w * p
            wsum += w
        z["score"] = (tot / wsum).clip(0, 100)
        return z


class SuitabilityMap(CommercialMap):
    def __init__(self, ctx, profile: str = "retail"):
        super().__init__(ctx)
        self.prof = PROFILES[profile]
        self.key = f"suitability_{profile}"

    def build(self) -> Page:
        c, b, pr = self.c, self.c.brand, self.prof
        end = c.dataset.window_label().split("–")[-1].strip()
        pg = Page(b, pr.title, f"Houston metro  ·  {pr.blurb}", SRC.format(end=end), c.stamp)
        ax = pg.map_axes((0.03, 0.07, 0.64, 0.835), (c.site.xy[0] + 3000, c.site.xy[1] - 4000), 98_000)
        z = SuitabilityModel(c).score(pr)
        base = BaseMap(c, pg)
        base.water_land(ax)
        edges = np.linspace(z["score"].min(), z["score"].max(), 6)
        cols = PaletteLibrary.OCEAN.colors(5, 0.12, 1.0) if pr.key == "retail" else PaletteLibrary.EMBER.colors(5, 0.1, 1.0)
        z["cls"] = np.clip(np.searchsorted(edges[1:-1], z["score"], side="right"), 0, 4)
        with pg.layer("suitability-choropleth", ax):
            nd = c.zips[~c.zips["zip"].isin(z["zip"])]
            nd.plot(ax=ax, fc="#E4E6E9", ec="white", hatch="////", lw=0.3, zorder=2)
            for i, col in enumerate(cols):
                z[z["cls"] == i].plot(ax=ax, fc=col, ec="white", lw=0.5, zorder=2.2)
        base.roads(ax, scale=0.9)
        pg.site_pin(ax, c.site.xy, c.site.name)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        pan = pg.panel((0.69, 0.07, 0.28, 0.835), "Suitability score")
        y = 0.89
        for i in range(4, -1, -1):
            pan.add_patch(Rectangle((0.06, y - 0.016), 0.09, 0.032, fc=cols[i], ec="white", transform=pan.transAxes))
            pan.text(0.19, y, f"{LABELS[i]}  ({edges[i]:.0f}–{edges[i + 1]:.0f})", fontsize=7, color=b.ink, va="center", transform=pan.transAxes)
            y -= 0.045
        y -= 0.025
        pan.text(0.06, y, "WEIGHTS", fontsize=6.2, fontweight="bold", color=b.primary, transform=pan.transAxes)
        y -= 0.04
        for k, w in sorted(pr.weights.items(), key=lambda t: -t[1]):
            pan.text(0.06, y, FACTORS[k].label, fontsize=6, color=b.ink, va="center", transform=pan.transAxes)
            pan.add_patch(Rectangle((0.06, y - 0.027), 0.88 * w / 0.4, 0.011, fc=b.secondary, ec="none", transform=pan.transAxes))
            pan.text(0.94, y, f"{w:.0%}", fontsize=6.4, fontweight="bold", color=b.ink, ha="right", va="center", transform=pan.transAxes)
            y -= 0.062
        y -= 0.01
        pan.text(0.06, y, "TOP 5 ZIPS", fontsize=6.2, fontweight="bold", color=b.primary, transform=pan.transAxes)
        y -= 0.04
        for _, r in z.sort_values("score", ascending=False).head(5).iterrows():
            pan.text(0.06, y, r["zip"], fontsize=7.2, fontweight="bold", color=b.ink, va="center", transform=pan.transAxes)
            pan.text(0.94, y, f"{r['score']:.0f}", fontsize=7.4, fontweight="bold", color=b.primary, ha="right", va="center", transform=pan.transAxes)
            y -= 0.036
        return pg
