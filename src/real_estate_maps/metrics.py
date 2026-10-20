"""What can be mapped. Each Metric knows its column, wording, number format, palette and classification scheme."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .palettes import Palette, PaletteLibrary


class Fmt:
    """Number formatters (static; used by legends, labels and the explorer)."""

    @staticmethod
    def money(v: float) -> str:
        return f"${v:,.0f}"

    @staticmethod
    def money_short(v: float) -> str:
        if abs(v) >= 1e6:
            return f"${v / 1e6:.1f}M".replace(".0M", "M")
        return f"${v / 1e3:,.0f}K" if abs(v) >= 1e3 else f"${v:,.0f}"

    @staticmethod
    def ppsf(v: float) -> str:
        return f"${v:,.0f}"

    @staticmethod
    def pct(v: float) -> str:
        return f"{v * 100:.0f}%"

    @staticmethod
    def pct_signed(v: float) -> str:
        r = round(v * 100)
        return "0%" if r == 0 else f"{r:+d}%"

    @staticmethod
    def days(v: float) -> str:
        return f"{v:,.0f}"

    @staticmethod
    def count(v: float) -> str:
        return f"{v:,.0f}"

    @staticmethod
    def ratio(v: float) -> str:
        return f"{v * 100:.1f}%"


@dataclass(frozen=True)
class Metric:
    key: str
    column: str
    title: str                         # big map title
    subtitle: str                      # what the number is
    legend_title: str
    fmt: Callable[[float], str]
    short: Callable[[float], str]
    palette: Palette
    scheme: str = "quantile"           # quantile | natural | fixed
    classes: int = 6
    breaks: tuple[float, ...] | None = None   # interior breaks for `fixed`
    min_sales: int = 10                # ZIPs with fewer sales in the window are shown as 'insufficient data'
    higher_is: str = ""                # plain-language reading help for the legend footnote
    note: str = ""


class MetricRegistry:
    """The built-in metrics. Add your own with `MetricRegistry.register(Metric(...))` - see docs/map-elements.md."""

    _m: dict[str, Metric] = {}

    @classmethod
    def register(cls, m: Metric) -> Metric:
        cls._m[m.key] = m
        return m

    @classmethod
    def get(cls, key: str) -> Metric:
        if key not in cls._m:
            raise KeyError(f"unknown metric '{key}'. Available: {', '.join(sorted(cls._m))}")
        return cls._m[key]

    @classmethod
    def keys(cls) -> list[str]:
        return list(cls._m)


MetricRegistry.register(Metric("price", "median_sale_price", "HOUSTON HOME PRICES", "Median sale price by ZIP code", "Median sale price", Fmt.money, Fmt.money_short,
                               PaletteLibrary.OCEAN, "natural", 7, higher_is="Darker = more expensive"))
MetricRegistry.register(Metric("ppsf", "median_ppsf", "PRICE PER SQUARE FOOT", "Median sale price per sq ft by ZIP code", "Median $ per sq ft", Fmt.ppsf, Fmt.ppsf,
                               PaletteLibrary.EMBER, "natural", 7, higher_is="Darker = more expensive per sq ft"))
MetricRegistry.register(Metric("yoy", "price_yoy", "WHERE PRICES ARE MOVING", "Change in median sale price vs. the same window a year earlier", "Year-over-year change", Fmt.pct_signed, Fmt.pct_signed,
                               PaletteLibrary.DIVERGING, "fixed", 8, breaks=(-0.15, -0.08, -0.03, 0.0, 0.03, 0.08, 0.15), higher_is="Blue = rising, red = falling",
                               note="Small ZIPs are volatile; a 3-month window can swing widely."))
MetricRegistry.register(Metric("dom", "median_dom", "HOW FAST HOMES SELL", "Median days on market by ZIP code", "Median days on market", Fmt.days, Fmt.days,
                               PaletteLibrary.ORCHID, "natural", 6, higher_is="Darker = slower sales"))
MetricRegistry.register(Metric("above_list", "sold_above_list", "BIDDING WARS", "Share of homes that sold above their list price", "Sold above list", Fmt.pct, Fmt.pct,
                               PaletteLibrary.MOSS, "natural", 6, higher_is="Darker = more competition"))
MetricRegistry.register(Metric("sale_to_list", "sale_to_list", "WHAT BUYERS ACTUALLY PAY", "Average sale price as a share of list price", "Sale-to-list ratio", Fmt.ratio, Fmt.ratio,
                               PaletteLibrary.SLATE, "natural", 6, higher_is="Darker = closer to (or above) asking"))
MetricRegistry.register(Metric("sales", "homes_sold", "WHERE THE SALES HAPPEN", "Homes sold in the 3-month window", "Homes sold", Fmt.count, Fmt.count,
                               PaletteLibrary.SLATE, "natural", 6, min_sales=0, higher_is="Darker = more sales"))
