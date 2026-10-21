"""Investor pages built on the walk-forward forecast: forecast map, risk-return map and an evidence page."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

from ..palettes import PaletteLibrary
from .base import BaseMap, CommercialMap, MapContext
from .inputs import to_xy
from .page import Page, halo

DOWNTOWN = (-95.3698, 29.7604)


class InvestorData:
    """Forecast table + polygons + backtest summary, shared by the three pages."""

    def __init__(self, ctx: MapContext):
        f = ctx.paths.processed / "forecast"
        self.fc = pd.read_csv(f / "zip_forecast.csv", dtype={"zip": str})
        self.bt = json.loads((f / "backtest_summary.json").read_text())
        names = pd.read_csv(ctx.paths.raw / "rdc" / "houston_core.csv", dtype={"postal_code": str}, usecols=["postal_code", "zip_name"]).drop_duplicates("postal_code", keep="last")
        names["area"] = names["zip_name"].str.replace(r"^\d{5},?\s*", "", regex=True).str.replace(r",?\s*tx$", "", regex=True, case=False).str.title()
        self.fc = self.fc.merge(names[["postal_code", "area"]].rename(columns={"postal_code": "zip"}), on="zip", how="left")
        for h in (12, 24):
            for k in ("rel", "lo", "hi"):
                self.fc[f"{k}_{h}_pct"] = np.expm1(self.fc[f"{k}_{h}"]) * 100
            self.fc[f"width_{h}_pct"] = self.fc[f"hi_{h}_pct"] - self.fc[f"lo_{h}_pct"]
        self.gdf = ctx.zips[["zip", "geometry"]].merge(self.fc, on="zip", how="left")
        self.origin = self.bt["forecast_origin"]
        self.model = self.bt["horizons"]["24"]["models"][self.bt["final_set"]]["investable"]
        self.model12 = self.bt["horizons"]["12"]["models"][self.bt["final_set"]]["investable"]

    def picks(self, h: int = 24, n: int = 10) -> pd.DataFrame:
        return self.fc[self.fc["investable"]].sort_values(f"rel_{h}", ascending=False).head(n)


def _stamp(c: MapContext) -> str | None:
    return "SAMPLE BRAND KIT — illustrative" if c.brand.sample else None


SRC_COMMON = ("Method: gradient-boosted trees on ZIP-level Redfin sales dynamics and neighboring-ZIP spillover, trained only on outcomes already known at each forecast date (walk-forward, 2020–{ly}). "
              "Forecasts are statistical rankings, not guarantees; they use housing-market data, not rents, zoning, permits or FEMA flood zones. ZIP areas are Census 2010 ZCTAs.")


class ForecastMap(CommercialMap):
    """Expected outperformance by ZIP, with low-confidence ZIPs hatched and the ten best investable picks numbered."""

    EDGES = [-99, -8, -4, -1.5, 1.5, 4, 8, 99]

    def __init__(self, ctx: MapContext, horizon: int = 24):
        super().__init__(ctx)
        self.h = horizon
        self.key = f"investor_forecast_{horizon}m"

    def build(self) -> Page:
        c, b, h = self.c, self.c.brand, self.h
        D = InvestorData(c)
        ly = D.bt["horizons"][str(h)]["years"][-1]
        pg = Page(b, "Houston Investment Forecast", f"Expected {h}-month price-per-sq-ft growth vs the typical ZIP  ·  forecast from {pd.Period(D.origin).strftime('%B %Y')}",
                  SRC_COMMON.format(ly=ly), _stamp(c))
        ax = pg.map_axes((0.03, 0.07, 0.60, 0.835), to_xy(-95.42, 29.78), 112_000)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        g = D.gdf.copy()
        col = f"rel_{h}_pct"
        pal = PaletteLibrary.DIVERGING.colors(7)
        g["cls"] = np.where(g[col].notna(), np.clip(np.searchsorted(self.EDGES[1:-1], g[col], side="right"), 0, 6), -1)
        with pg.layer("forecast-choropleth", ax):
            for i, colr in enumerate(pal):
                for inv, alpha in ((True, 1.0), (False, 0.42)):
                    sub = g[(g["cls"] == i) & (g["investable"] == inv)]
                    if len(sub):
                        sub.plot(ax=ax, fc=colr, ec="white", lw=0.5, alpha=alpha, zorder=2)
            g[g["cls"] < 0].plot(ax=ax, fc="#E4E6E9", ec="white", lw=0.4, zorder=2)
        wcol = f"width_{h}_pct"
        thr = g.loc[g["investable"] == True, wcol].quantile(2 / 3)
        with pg.layer("low-confidence-hatch", ax):
            wide = g[(g[wcol] > thr) & g["investable"]]
            if len(wide):
                wide.plot(ax=ax, fc="none", ec="#1B2430", hatch="////", lw=0, alpha=0.35, zorder=3)
        base.roads(ax, label=False, scale=0.55)
        picks = D.picks(h, 10).reset_index(drop=True)
        gi = g.set_index("zip")
        with pg.layer("top-picks", ax):
            for i, r in picks.iterrows():
                pt = gi.loc[r["zip"], "geometry"].representative_point()
                pg.badge(ax, (pt.x, pt.y), str(i + 1), color=b.primary, r_frac=0.011, fs=6.8, zorder=30)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        # legend (inside map, bottom right)
        x0, y0, x1, y1 = pg.extent(ax)
        W, H = x1 - x0, y1 - y0
        lx, ly_ = x1 - W * 0.30, y0 + H * 0.045
        with pg.layer("legend", ax):
            ax.add_patch(Rectangle((lx - W * 0.012, ly_ - H * 0.02), W * 0.31, H * 0.245, fc="white", alpha=0.9, ec="none", zorder=40))
            ax.text(lx, ly_ + H * 0.2, f"EXPECTED {h}-MO GROWTH vs TYPICAL ZIP", fontsize=5.8, fontweight="bold", color=b.primary, zorder=41)
            labs = ["below -8%", "-8 to -4%", "-4 to -1.5%", "within ±1.5%", "+1.5 to +4%", "+4 to +8%", "above +8%"]
            for i, (cc, ll) in enumerate(zip(pal, labs)):
                ax.add_patch(Rectangle((lx + i * W * 0.04, ly_ + H * 0.12), W * 0.04, H * 0.04, fc=cc, ec="white", zorder=41))
            ax.text(lx, ly_ + H * 0.105, "weaker", fontsize=5.2, color=b.muted, va="top", zorder=41)
            ax.text(lx + 7 * W * 0.04, ly_ + H * 0.105, "stronger", fontsize=5.2, color=b.muted, va="top", ha="right", zorder=41)
            ax.add_patch(Rectangle((lx, ly_ + H * 0.045), W * 0.035, H * 0.03, fc="#9AA5B1", ec="#1B2430", hatch="////", lw=0.2, zorder=41))
            ax.text(lx + W * 0.045, ly_ + H * 0.06, "wider forecast range (least certain third)", fontsize=5.4, color=b.ink, va="center", zorder=41)
            ax.add_patch(Rectangle((lx, ly_ + H * 0.0), W * 0.035, H * 0.03, fc="#9AA5B1", alpha=0.42, ec="none", zorder=41))
            ax.text(lx + W * 0.045, ly_ + H * 0.015, "thin market (<50 sales per quarter)", fontsize=5.4, color=b.ink, va="center", zorder=41)
        # right panel: shortlist
        pan = pg.panel((0.645, 0.07, 0.325, 0.835), f"Top 10 investable ZIPs · {h}-month outlook")
        pan.text(0.04, 0.905, "ZIP / area", fontsize=5.6, color=b.muted, fontweight="bold", va="center", transform=pan.transAxes)
        pan.text(0.62, 0.905, "vs typical ZIP", fontsize=5.6, color=b.muted, fontweight="bold", va="center", ha="right", transform=pan.transAxes)
        pan.text(0.96, 0.905, "80% range", fontsize=5.6, color=b.muted, fontweight="bold", va="center", ha="right", transform=pan.transAxes)
        y = 0.865
        for i, r in picks.iterrows():
            pg.dot(pan, 0.055, y, str(i + 1), b.primary, 9, 5.6)
            pan.text(0.10, y + 0.006, f"{r['zip']}", fontsize=7.4, fontweight="bold", color=b.ink, va="center", transform=pan.transAxes)
            pan.text(0.10, y - 0.016, f"{str(r['area'])[:22]} · ${r['ppsf']:.0f}/sq ft · {r['sold_3mo']:.0f} sales", fontsize=5.2, color=b.muted, va="center", transform=pan.transAxes)
            pan.text(0.62, y, f"{r[f'rel_{h}_pct']:+.1f}%", fontsize=8.6, fontweight="bold", color=b.secondary, ha="right", va="center", family=b.heading, transform=pan.transAxes)
            pan.text(0.96, y, f"{r[f'lo_{h}_pct']:+.0f} to {r[f'hi_{h}_pct']:+.0f}%", fontsize=6.2, color=b.muted, ha="right", va="center", transform=pan.transAxes)
            y -= 0.041
        # backtest credibility box
        m = D.bt["horizons"][str(h)]["models"][D.bt["final_set"]]["investable"]
        dec = D.bt["horizons"][str(h)]["decile_realized_rel_growth"]["investable"]
        box = pg.fig.add_axes((0.655, 0.085, 0.305, 0.325))
        box.set_axis_off(); box.set_xlim(0, 1); box.set_ylim(0, 1)
        box.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec=b.primary, lw=0.6))
        box.text(0.04, 0.93, "DOES IT WORK? OUT-OF-TIME BACKTEST", fontsize=6.2, fontweight="bold", color=b.primary, va="center")
        box.text(0.04, 0.78, f"{m['top10'] * 100:+.1f}%", fontsize=17, fontweight="bold", family=b.heading, color=b.secondary, va="center")
        box.text(0.30, 0.80, f"avg {h}-month growth vs typical ZIP\nfor the model's top-decile picks", fontsize=5.8, color=b.ink, va="center", linespacing=1.3)
        box.text(0.04, 0.58, f"{m['bottom10'] * 100:+.1f}%", fontsize=13, fontweight="bold", family=b.heading, color=b.alert, va="center")
        box.text(0.30, 0.58, "for its bottom-decile ZIPs", fontsize=5.8, color=b.ink, va="center")
        box.text(0.04, 0.41, f"{m['precision20'] * 100:.0f}%", fontsize=13, fontweight="bold", family=b.heading, color=b.primary, va="center")
        box.text(0.30, 0.41, "of its top-fifth picks were truly in the top fifth\n(chance = 20%)", fontsize=5.8, color=b.ink, va="center", linespacing=1.3)
        box.text(0.04, 0.2, f"Tested on {m['origins']} monthly forecast dates, {D.bt['horizons'][str(h)]['years'][0]}–{D.bt['horizons'][str(h)]['years'][-1]};\nZIPs with ≥{D.bt['investable_min_sales']} sales per quarter; positive in {m['years_positive']} of {m['years']} years.\n"
                 "An edge that tilts the odds, not a guarantee.", fontsize=5.4, color=b.muted, va="center", linespacing=1.35)
        return pg


BIVAR = [["#E8E8E8", "#ACE4E4", "#5AC8C8"], ["#DFB0D6", "#A5ADD3", "#5698B9"], ["#BE64AC", "#8C62AA", "#3B4994"]]   # rows: certainty high->low ... see legend


class RiskReturnMap(CommercialMap):
    """Bivariate map: expected outperformance (x) against forecast uncertainty (y), with an annotated scatter and a payoff table."""

    key = "investor_risk_return"
    H = 24

    def build(self) -> Page:
        c, b, h = self.c, self.c.brand, self.H
        D = InvestorData(c)
        ly = D.bt["horizons"][str(h)]["years"][-1]
        pg = Page(b, "Risk vs Return", f"Where expected outperformance is high and the forecast is firm  ·  {h}-month outlook from {pd.Period(D.origin).strftime('%B %Y')}", SRC_COMMON.format(ly=ly), _stamp(c))
        ax = pg.map_axes((0.03, 0.07, 0.56, 0.835), to_xy(-95.42, 29.78), 100_000)
        base = BaseMap(c, pg)
        base.water_land(ax, counties=False)
        g = D.gdf.copy()
        inv = g[g["investable"] == True].copy()
        xq = inv[f"rel_{h}_pct"].quantile([1 / 3, 2 / 3]).to_numpy()
        yq = inv[f"width_{h}_pct"].quantile([1 / 3, 2 / 3]).to_numpy()
        inv["xc"] = np.searchsorted(xq, inv[f"rel_{h}_pct"], side="right")
        inv["yc"] = np.searchsorted(yq, inv[f"width_{h}_pct"], side="right")
        with pg.layer("bivariate-choropleth", ax):
            g[g["investable"] != True].plot(ax=ax, fc="#EEF0F2", ec="white", lw=0.4, zorder=2)
            for (xc, yc), sub in inv.groupby(["xc", "yc"]):
                sub.plot(ax=ax, fc=BIVAR[yc][xc], ec="white", lw=0.5, zorder=2.2)
        base.roads(ax, label=False, scale=0.5)
        prime = inv[(inv["xc"] == 2) & (inv["yc"] == 0)].sort_values(f"rel_{h}_pct", ascending=False)
        x0_, y0_, x1_, y1_ = pg.extent(ax)
        prime = prime[[x0_ < p.x < x1_ and y0_ < p.y < y1_ for p in prime.geometry.representative_point()]]
        with pg.layer("prime-targets", ax):
            for i, (_, r) in enumerate(prime.iterrows(), 1):
                pt = r.geometry.representative_point()
                pg.badge(ax, (pt.x, pt.y), str(i), color=b.alert, r_frac=0.011, fs=6.5, zorder=30)
            low = inv[inv["elev_p10_m"] < 8]
            for _, r in low.iterrows():
                pt = r.geometry.representative_point()
                ax.plot(pt.x, pt.y, marker="v", ms=4.5, mfc="#1B2430", mec="white", mew=0.5, zorder=29)
        pg.north_arrow(ax)
        pg.scale_bar(ax)
        # legend matrix (own white box, bottom-left of the map)
        lg = pg.fig.add_axes((0.04, 0.085, 0.225, 0.15))
        lg.set_xlim(-2.6, 6.8); lg.set_ylim(-1.5, 4.3); lg.set_axis_off()
        lg.add_patch(Rectangle((-2.6, -1.5), 9.4, 5.8, fc="white", alpha=0.95, ec="none"))
        for xi in range(3):
            for yi in range(3):
                lg.add_patch(Rectangle((xi, 2 - yi), 1, 1, fc=BIVAR[yi][xi], ec="white", lw=1.2))
        lg.text(1.5, -0.15, "expected outperformance  →", ha="center", va="top", fontsize=5.2, color=b.ink)
        lg.text(-1.55, 1.5, "forecast uncertainty  →", rotation=90, ha="center", va="center", fontsize=5.2, color=b.ink)
        lg.text(-0.15, 2.5, "firm", ha="right", va="center", fontsize=4.8, color=b.muted); lg.text(-0.15, 0.5, "wide", ha="right", va="center", fontsize=4.8, color=b.muted)
        lg.text(0.5, 3.1, "lower", ha="center", fontsize=4.8, color=b.muted); lg.text(2.5, 3.1, "higher", ha="center", fontsize=4.8, color=b.muted)
        lg.add_patch(Circle((3.85, 2.55), 0.22, fc=b.alert, ec="white"))
        lg.text(4.2, 2.55, "prime target:\nhigh return,\nfirm forecast", fontsize=4.9, color=b.ink, va="center", linespacing=1.3)
        lg.plot([3.85], [0.9], marker="v", ms=4, color="#1B2430", ls="none")
        lg.text(4.2, 0.9, "low-lying ZIP\n(p10 elev < 8 m):\ncheck flood risk", fontsize=4.9, color=b.ink, va="center", linespacing=1.3)
        # right column: scatter, prime list, payoff
        sc = pg.fig.add_axes((0.655, 0.615, 0.305, 0.225))
        sc.scatter(inv[f"rel_{h}_pct"], inv[f"width_{h}_pct"], s=9, c=[BIVAR[a_][b_] for a_, b_ in zip(inv["yc"], inv["xc"])], edgecolors="#1B2430", linewidths=0.3, zorder=3)
        sc.scatter(prime[f"rel_{h}_pct"], prime[f"width_{h}_pct"], s=22, facecolors="none", edgecolors=b.alert, linewidths=1.0, zorder=4)
        sc.axvline(xq[1], color=b.muted, lw=0.5, ls="--"); sc.axhline(yq[0], color=b.muted, lw=0.5, ls="--")
        sc.set_xlabel("expected 24-mo growth vs typical ZIP (%)", fontsize=5.6, color=b.ink); sc.set_ylabel("80% range width (pts)", fontsize=5.6, color=b.ink)
        sc.tick_params(labelsize=5.3, length=2, colors=b.muted)
        for s_ in ("top", "right"):
            sc.spines[s_].set_visible(False)
        sc.set_title("Return against certainty (red ring = prime target)", fontsize=6.4, fontweight="bold", color=b.primary, loc="left")
        pl = pg.panel((0.61, 0.335, 0.36, 0.215), "Prime targets in view")
        for i, (_, r) in enumerate(prime.head(11).iterrows(), 1):
            col_, row_ = (i - 1) // 6, (i - 1) % 6
            x0_, yy = 0.04 + col_ * 0.5, 0.775 - row_ * 0.125
            pg.dot(pl, x0_ + 0.025, yy, str(i), b.alert, 8, 5)
            pl.text(x0_ + 0.075, yy + 0.02, f"{r['zip']}  {r[f'rel_{h}_pct']:+.1f}%", fontsize=6.2, fontweight="bold", color=b.ink, va="center", transform=pl.transAxes)
            pl.text(x0_ + 0.075, yy - 0.04, f"{str(r['area'])[:16]} · {r['sold_3mo']:.0f} sales", fontsize=4.8, color=b.muted, va="center", transform=pl.transAxes)
        pan = pg.panel((0.61, 0.07, 0.36, 0.25))
        pan.text(0.04, 0.92, "ILLUSTRATIVE 24-MONTH VALUE OF $100,000", fontsize=6.0, fontweight="bold", color=b.primary, va="center", transform=pan.transAxes)
        m = D.model
        hist = D.bt["metro_ppsf_history"]
        scen = [("Metro\nflat", 0.0), ("Metro\n+2%/yr", 0.0404), (f"Metro +{hist['cagr_full'] * 100:.1f}%/yr\n(2012–26 avg)", (1 + hist["cagr_full"]) ** 2 - 1)]
        rows = [("Top-decile picks", m["top10"]), ("Typical ZIP", 0.0), ("Bottom-decile picks", m["bottom10"])]
        for j, (nm, _) in enumerate(scen):
            pan.text(0.50 + j * 0.2, 0.77, nm, fontsize=4.9, color=b.muted, ha="center", va="center", fontweight="bold", transform=pan.transAxes, linespacing=1.2)
        for i, (nm, a_) in enumerate(rows):
            yy = 0.58 - i * 0.16
            pan.text(0.04, yy, f"{nm}\n({a_ * 100:+.1f}% vs typical)", fontsize=5.3, color=b.ink, va="center", linespacing=1.3, transform=pan.transAxes)
            for j, (_, gs) in enumerate(scen):
                v = 100000 * (1 + gs) * np.exp(a_)
                pan.text(0.50 + j * 0.2, yy, f"${v:,.0f}", fontsize=6.6, fontweight="bold", color=b.secondary if a_ > 0 else (b.alert if a_ < 0 else b.ink), ha="center", va="center", family=b.heading, transform=pan.transAxes)
        import textwrap
        note = (f"Price appreciation only; excludes rent, costs, taxes, financing. Houston's median price per sq ft is up {hist['growth_last_36m'] * 100:+.1f}% over 36 months "
                f"({hist['growth_last_12m'] * 100:+.1f}% over 12), so the metro trend, not ZIP selection, dominates absolute returns. Relative figures are backtested group averages.")
        pan.text(0.04, 0.115, textwrap.fill(note, 95), fontsize=4.7, color=b.muted, va="center", linespacing=1.35, transform=pan.transAxes)
        return pg


class EvidencePage(CommercialMap):
    """The analysis behind the forecast: does it beat simple rules, what drives it, how reliable are its ranges."""

    key = "investor_evidence"
    H = 24

    def build(self) -> Page:
        c, b, h = self.c, self.c.brand, self.H
        D = InvestorData(c)
        bt = D.bt
        d = bt["horizons"][str(h)]
        ly = d["years"][-1]
        pg = Page(b, "Forecast Evidence", f"How the Houston ZIP forecast was built and tested  ·  {h}-month horizon, ZIPs with ≥{bt['investable_min_sales']} sales per quarter",
                  SRC_COMMON.format(ly=ly) + " Sources: Redfin Data Center (ZIP market tracker, 2012–2026), Realtor.com Research (listings 2016+, Hotness 2017+), Overture Maps (420k places, 885k road segments), USGS 3DEP elevation.", _stamp(c))
        ink, mut = b.ink, b.muted

        def style(a, title):
            a.set_title(title, fontsize=7.2, fontweight="bold", color=b.primary, loc="left", pad=6)
            a.tick_params(labelsize=5.8, length=2, colors=mut)
            for s_ in ("top", "right"):
                a.spines[s_].set_visible(False)
            a.set_facecolor("white")
        # 1: deciles
        a1 = pg.fig.add_axes((0.06, 0.55, 0.40, 0.30))
        dec = d["decile_realized_rel_growth"]["investable"]
        xs = np.arange(1, 11)
        vals = np.array([dec[str(i)] * 100 for i in xs])
        a1.bar(xs, vals, color=[b.alert if v < 0 else b.secondary for v in vals], width=0.7)
        a1.axhline(0, color=ink, lw=0.6)
        for x, v in zip(xs, vals):
            a1.text(x, v + (0.12 if v >= 0 else -0.12), f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=5.6, color=ink)
        a1.set_ylim(vals.min() - 0.7, vals.max() + 0.7); a1.set_xticks(xs); a1.set_xlabel("ZIPs grouped by the model's predicted growth (1 = lowest, 10 = highest)", fontsize=5.8, color=mut)
        a1.set_ylabel("actual 24-mo growth vs typical ZIP (%)", fontsize=5.8, color=mut)
        style(a1, "1  Predictions rank outcomes in order, out of time")
        # 2: skill bars
        a2 = pg.fig.add_axes((0.60, 0.55, 0.36, 0.30))
        items = [(k, v["investable"]["rho"], False) for k, v in d["baselines"].items() if k not in ("above own 12-mo trend",)]
        name_map = {"B_sales_market_neighbors": "Model (sales + neighbors)  ← used", "D_plus_hotness": "+ Realtor.com Hotness (not available now)", "E_plus_static_context": "+ amenity / road / elevation features",
                    "A_trailing_growth_only": "Model on trailing growth alone"}
        items += [(name_map[k], d["models"][k]["investable"]["rho"], True) for k in name_map]
        items.sort(key=lambda t: t[1])
        a2.barh([i[0] for i in items], [i[1] for i in items], color=[b.primary if "used" in i[0] else (b.secondary if i[2] else "#98A1AB") for i in items], height=0.62)
        a2.axvline(0, color=ink, lw=0.6); a2.set_xlim(-0.13, 0.27)
        for yy, (_, v, _) in enumerate(items):
            a2.text(v + (0.005 if v >= 0 else -0.005), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=5.6, color=ink)
        a2.tick_params(axis="y", labelsize=5.6)
        a2.set_xlabel("average rank correlation with actual outcomes (0 = no skill)", fontsize=5.8, color=mut)
        style(a2, "2  Skill against simple rules of thumb")
        a2.tick_params(axis="y", labelsize=5.6, colors=ink)
        # 3: by year
        a3 = pg.fig.add_axes((0.06, 0.09, 0.30, 0.30))
        mb = d["models"][bt["final_set"]]["investable"]["rho_by_year"]
        mm = d["baselines"]["trailing 12-mo growth (momentum)"]["investable"]["rho_by_year"]
        yrs = sorted(mb)
        a3.plot([int(y) for y in yrs], [mb[y] for y in yrs], marker="o", ms=3.5, color=b.primary, lw=1.8)
        a3.plot([int(y) for y in yrs], [mm[y] for y in yrs], marker="o", ms=3.5, color="#98A1AB", lw=1.4, ls="--")
        a3.axhline(0, color=ink, lw=0.6)
        a3.text(int(yrs[-1]) + 0.05, mb[yrs[-1]], "model", fontsize=5.8, color=b.primary, fontweight="bold", va="center")
        a3.text(int(yrs[-1]) + 0.05, mm[yrs[-1]], "just follow\nrecent growth", fontsize=5.4, color=mut, va="center")
        a3.set_xlim(int(yrs[0]) - 0.2, int(yrs[-1]) + 1.2); a3.set_xticks([int(y) for y in yrs])
        a3.set_ylabel("rank correlation", fontsize=5.8, color=mut)
        style(a3, "3  Consistency by test year")
        # 4: importance
        a4 = pg.fig.add_axes((0.45, 0.09, 0.23, 0.30))
        imp = d["permutation_importance"]
        lab = {"momentum": "price level &\ntrend vs metro", "market_tightness": "days on market,\nsupply, sales", "neighbors": "neighboring\nZIPs", "listings_rdc": "Realtor.com\nlistings"}
        ks = sorted(imp, key=imp.get)
        a4.barh([lab.get(k, k) for k in ks], [imp[k] for k in ks], color=b.secondary, height=0.6)
        a4.axvline(0, color=ink, lw=0.6)
        a4.set_xlabel("skill lost when the group is shuffled", fontsize=5.8, color=mut)
        style(a4, "4  What drives the forecast")
        a4.tick_params(axis="y", labelsize=5.6, colors=ink)
        # findings panel
        pan = pg.panel((0.715, 0.07, 0.255, 0.38), "What the analysis found")
        mom = d["baselines"]["trailing 12-mo growth (momentum)"]["investable"]
        raw, cov = d["interval_coverage_80_raw"]["investable"], d["interval_coverage_80"]["investable"]
        allr, invr = d["models"][bt["final_set"]]["all"]["rho"], d["models"][bt["final_set"]]["investable"]["rho"]
        sp = [x for x in d["spatial_holdout"] if x["zips_in_block"] >= 10]
        lines = [f"Hot ZIPs cool off. Rank correlation of trailing growth with the next 24 months is {mom['rho']:+.2f}; positive in {mom['years_positive']} of {mom['years']} years.",
                 "Cheap next to richer neighbors tends to catch up (the ripple effect); expensive ZIPs that ran ahead tend to lag.",
                 f"Realtor.com listings added nothing here ({d['models']['C_plus_listings']['investable']['rho']:+.2f} vs {invr:+.2f}); amenity, road and elevation features hurt out of time, so they are used for risk flags only.",
                 f"Held-out geography: skill stays positive in {sum(x['rho'] > 0 for x in sp)} of {len(sp)} regions not used for training.",
                 f"Median noise inflates headline skill: {allr:+.2f} on all ZIPs vs {invr:+.2f} on liquid ones. The liquid number is the one to trust.",
                 f"Ranges: raw 80% bands covered {raw * 100:.0f}% of outcomes; rescaled by {d['interval_scale']:.2f}x they cover {cov * 100:.0f}%."]
        y = 0.86
        import textwrap
        for ln in lines:
            t = textwrap.fill(ln, 46)
            pan.text(0.05, y, "• " + t.replace("\n", "\n   "), fontsize=5.5, color=ink, va="top", linespacing=1.32, transform=pan.transAxes)
            y -= 0.04 + 0.034 * t.count("\n") + 0.03
        return pg
