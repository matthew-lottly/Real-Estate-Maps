"""Named colour ramps (hand-picked anchors, interpolated to any number of classes)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex


@dataclass(frozen=True)
class Palette:
    name: str
    anchors: tuple[str, ...]
    diverging: bool = False

    def cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(self.name, list(self.anchors))

    def colors(self, n: int, lo: float = 0.0, hi: float = 1.0) -> list[str]:
        return [to_hex(self.cmap()(v)) for v in np.linspace(lo, hi, n)]

    def reversed(self) -> "Palette":
        return Palette(self.name + "_r", tuple(reversed(self.anchors)), self.diverging)


class PaletteLibrary:
    OCEAN = Palette("ocean", ("#F6F1E4", "#DCE8D2", "#AAD5C0", "#6DB9B1", "#3C93A4", "#2B6D94", "#264A82", "#1C2A5C"))
    EMBER = Palette("ember", ("#FCF3E0", "#F8DDAF", "#F0BB78", "#E5944E", "#D06B3B", "#A8442F", "#6F2527"))
    MOSS = Palette("moss", ("#F5F6E8", "#DDE9B8", "#B5D58B", "#80BA6B", "#4C9A5A", "#2A7449", "#16482F"))
    ORCHID = Palette("orchid", ("#F6F1F9", "#DDD2EC", "#B9A7D8", "#9179BF", "#6C55A2", "#483180", "#2B1B55"))
    SLATE = Palette("slate", ("#F2F4F6", "#D6DCE3", "#AEB9C6", "#8497AB", "#5D748D", "#3C536C", "#223347"))
    DIVERGING = Palette("rdbu", ("#B2182B", "#D6604D", "#F4A582", "#FBDCCB", "#F4F4F4", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC"), True)

    @classmethod
    def get(cls, name: str) -> Palette:
        try:
            return getattr(cls, name.upper())
        except AttributeError as exc:
            raise KeyError(f"unknown palette '{name}'; available: {[n for n in dir(cls) if n.isupper()]}") from exc
