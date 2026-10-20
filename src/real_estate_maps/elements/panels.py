from __future__ import annotations

import textwrap

import geopandas as gpd
import numpy as np
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import box
from shapely.ops import unary_union

from ..composer import MapElement, RenderContext
from .rankings import Rankings


def _txt(ax, x, y, s, **kw):
    kw.setdefault("transform", ax.transAxes)
    return ax.text(x, y, s, **kw)


class TitleBlock(MapElement):
    name = "title"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("title")
        t = c.theme
        _txt(a, 0, 0.97, "REAL ESTATE MAP  ·  HOUSTON METRO", fontsize=9.5, fontweight="bold", color=t.accent, va="top")
        _txt(a, 0, 0.84, c.metric.title, fontsize=min(30.0, 4.5 * 72 / (0.68 * len(c.metric.title))), fontweight="heavy", color=t.ink, va="top", linespacing=0.95)
        _txt(a, 0, 0.40, c.metric.subtitle, fontsize=11.5, color=t.muted, va="top", wrap=True)
        a.add_patch(Rectangle((0, 0.03), 0.34, 0.012, transform=a.transAxes, facecolor=t.accent, edgecolor="none"))
        _txt(a, 0.0, 0.075, f"3 months ending {c.period_label.split(chr(8211))[-1].strip()}", fontsize=10.5, fontweight="bold", color=t.ink, va="bottom")


class StatsPanel(MapElement):
    name = "stats"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("stats")
        t, m, z = c.theme, c.metric, c.zips
        vals = z.loc[z["eligible"], m.column].dropna()
        items = [("TYPICAL ZIP", m.fmt(float(vals.median())), "median of ZIP values"),
                 ("HIGHEST", m.fmt(float(vals.max())), z.loc[vals.idxmax(), "zip"]),
                 ("LOWEST", m.fmt(float(vals.min())), z.loc[vals.idxmin(), "zip"])]
        for i, (lab, val, sub) in enumerate(items):
            x = i / 3
            _txt(a, x, 0.95, lab, fontsize=8, fontweight="bold", color=t.muted, va="top")
            _txt(a, x, 0.62, val, fontsize=15, fontweight="heavy", color=t.ink, va="top")
            _txt(a, x, 0.12, sub, fontsize=8, color=t.faint, va="top")


class LegendBlock(MapElement):
    """Stepped colour bar with the number of ZIPs in each class, plus the 'no data' swatch."""

    name = "legend"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("legend")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        t, s, m = c.theme, c.scheme, c.metric
        _txt(a, 0, 0.98, m.legend_title.upper(), fontsize=9, fontweight="bold", color=t.ink, va="top")
        _txt(a, 0, 0.88, f"{m.higher_is}  ·  bars show ZIP count", fontsize=8.5, color=t.muted, va="top")
        n, w = s.n, 1.0 / s.n
        counts = s.counts(c.zips.loc[c.zips["eligible"], m.column].to_numpy())
        top = max(counts) or 1
        for i, col in enumerate(s.colors):
            a.add_patch(Rectangle((i * w, 0.42), w, 0.16, facecolor=col, edgecolor="white", linewidth=1.4, transform=a.transAxes))
            h = 0.11 * counts[i] / top
            a.add_patch(Rectangle((i * w + w * 0.18, 0.62), w * 0.64, max(h, 0.004), facecolor=t.faint, edgecolor="none", alpha=0.6, transform=a.transAxes))
            _txt(a, i * w + w / 2, 0.62 + max(h, 0.004) + 0.012, str(counts[i]), fontsize=7.5, color=t.muted, ha="center", va="bottom")
        for i, e in enumerate(s.edges):
            _txt(a, i * w, 0.37, m.short(e), fontsize=7.8, color=t.ink, ha="center", va="top")
        n_nd = int((~c.zips["eligible"]).sum())
        a.add_patch(Rectangle((0, 0.04), 0.07, 0.13, facecolor=t.nodata, edgecolor=t.nodata_hatch, hatch="////", linewidth=0.5, transform=a.transAxes))
        _txt(a, 0.095, 0.105, f"Fewer than {m.min_sales} sales or no data ({n_nd} ZIPs)" if m.min_sales else f"No data ({n_nd} ZIPs)", fontsize=8.5, color=t.muted, va="center")


class RankingPanel(MapElement):
    name = "ranking"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("ranking")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        t, m = c.theme, c.metric
        top, bottom = Rankings.of(c)
        vmax = max(r.value for r in top + bottom) or 1
        vmin = min(0.0, min(r.value for r in top + bottom))
        rows = [("HIGHEST", top, True), ("LOWEST", bottom, False)]
        y = 0.98
        for title, items, filled in rows:
            _txt(a, 0, y, f"{title} ZIP CODES", fontsize=9, fontweight="bold", color=t.ink, va="top")
            y -= 0.07
            for r in items:
                col = c.scheme.colors[r.cls] if r.cls >= 0 else t.faint
                a.add_patch(Circle((0.025, y - 0.03), 0.022, transform=a.transAxes, facecolor=t.ink if filled else "white", edgecolor=t.ink, linewidth=1.2))
                _txt(a, 0.025, y - 0.03, str(r.rank), fontsize=7.5, fontweight="bold", color="white" if filled else t.ink, ha="center", va="center")
                _txt(a, 0.085, y - 0.03, r.zip, fontsize=9.5, fontweight="bold", color=t.ink, va="center")
                span = vmax - vmin or 1
                bw = 0.43 * (r.value - vmin) / span
                a.add_patch(Rectangle((0.25, y - 0.05), max(bw, 0.004), 0.036, transform=a.transAxes, facecolor=col, edgecolor=t.ink, linewidth=0.5))
                _txt(a, 0.98, y - 0.03, m.fmt(r.value), fontsize=9.5, color=t.ink, ha="right", va="center", fontweight="bold")
                y -= 0.072
            y -= 0.025


class InsetLocator(MapElement):
    """Texas with the mapped area outlined."""

    name = "inset"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("inset")
        tx = unary_union([g.buffer(0.002) for g in c.counties.geometry]).buffer(-0.002)
        gpd_tx = gpd.GeoSeries([tx], crs=c.counties.crs)
        gpd_tx.plot(ax=a, facecolor=c.theme.land, edgecolor=c.theme.ink, linewidth=0.8)
        minx, miny, maxx, maxy = c.extent
        a.add_patch(Rectangle((minx, miny), maxx - minx, maxy - miny, facecolor="none", edgecolor=c.theme.accent, linewidth=1.8))
        a.set_aspect("equal")
        a.set_axis_off()
        bx0, by0, bx1, by1 = tx.bounds
        c.fig.text(c.rects["inset"][0], c.rects["inset"][1] + c.rects["inset"][3] + 0.004, "WHERE IN TEXAS", fontsize=7.5, fontweight="bold", color=c.theme.muted, va="bottom")


class HowToRead(MapElement):
    name = "howto"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("howto")
        t, m = c.theme, c.metric
        notes = ["HOW TO READ", f"Each area is a ZIP code. Fill shows the {m.legend_title.lower()}.",
                 "ZIPs are postal areas (Census ZCTAs), not neighborhoods.", "Hatched ZIPs had too few sales to map reliably."]
        if m.note:
            notes.append(m.note)
        y = 1.0
        for i, s in enumerate(notes):
            s = textwrap.fill(s, 44)
            _txt(a, 0, y, s, fontsize=8 if i else 8.5, fontweight="bold" if i == 0 else "normal", color=t.ink if i == 0 else t.muted, va="top")
            y -= 0.14 if i == 0 else 0.07 + 0.075 * s.count("\n") + 0.07


class SourceNote(MapElement):
    name = "source"

    def draw(self, c: RenderContext) -> None:
        a = c.sub_axes("source")
        t = c.theme
        txt = (f"Data: Redfin Data Center, ZIP-code market tracker (all residential, rolling 3-month windows), updated {c.dataset.last_updated()}. "
               "Boundaries: US Census Bureau ZIP Code Tabulation Areas (2010) and county outlines. Newer ZIPs without a 2010 area are not drawn. "
               "Map: Real Estate Maps (github.com/matthew-lottly/Real-Estate-Maps).")
        _txt(a, 0, 1.0, txt, fontsize=6.8, color=t.faint, va="top", wrap=True)
