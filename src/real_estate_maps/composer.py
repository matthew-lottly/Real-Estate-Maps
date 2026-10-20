"""MapComposer: assembles a list of MapElements into a finished poster map."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pyproj import Transformer
from shapely.geometry import box

from .classification import ClassifierFactory, ClassScheme
from .config import HoustonFrame, Paths
from .data import MarketDataset
from .metrics import Metric
from .theme import MapTheme


@dataclass
class RenderContext:
    """Everything an element needs. Elements read it and draw; they never change the data."""

    fig: Figure
    ax: plt.Axes                       # the map axes (projected metres)
    theme: MapTheme
    metric: Metric
    scheme: ClassScheme
    zips: gpd.GeoDataFrame             # projected, with `cls` (class index, -1 = no data) and `eligible`
    counties: gpd.GeoDataFrame
    dataset: MarketDataset
    frame: HoustonFrame
    extent: tuple[float, float, float, float]
    period_label: str
    rects: dict[str, tuple[float, float, float, float]]      # named figure-fraction rectangles
    font: str
    to_xy: callable = None

    def sub_axes(self, name: str, **kw) -> plt.Axes:
        a = self.fig.add_axes(self.rects[name], **kw)
        a.set_axis_off()
        return a


class MapElement:
    """One piece of the map (title, legend, scale bar...). Override `draw`."""

    name = "element"

    def draw(self, c: RenderContext) -> None:
        raise NotImplementedError


class MapComposer:
    """Poster layout: a large map on the left, a sidebar on the right. Pass your own element list to rearrange or restyle."""

    SIZE = (17.0, 11.0)
    RECTS = {"map": (0.018, 0.035, 0.665, 0.93), "title": (0.705, 0.80, 0.275, 0.175), "stats": (0.705, 0.695, 0.275, 0.095),
             "legend": (0.705, 0.505, 0.275, 0.175), "ranking": (0.705, 0.215, 0.275, 0.275), "inset": (0.705, 0.075, 0.125, 0.125),
             "howto": (0.84, 0.075, 0.14, 0.125), "source": (0.705, 0.012, 0.275, 0.055)}

    def __init__(self, dataset: MarketDataset, metric: Metric, elements: list[MapElement], theme: MapTheme | None = None,
                 frame: HoustonFrame | None = None, period_end: str | None = None):
        self.ds, self.metric, self.elements = dataset, metric, elements
        self.theme, self.frame, self.period_end = theme or MapTheme(), frame or HoustonFrame(), period_end

    def _prepare(self) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, ClassScheme]:
        m = self.metric
        snap = self.ds.snapshot(self.period_end).to_crs(self.frame.crs)
        snap["eligible"] = snap[m.column].notna() & (snap["homes_sold"].fillna(0) >= m.min_sales)
        scheme = ClassifierFactory.for_metric(m).scheme(snap.loc[snap["eligible"], m.column].to_numpy())
        snap["cls"] = np.where(snap["eligible"], scheme.assign(snap[m.column].to_numpy()), -1)
        return snap, self.ds.counties.to_crs(self.frame.crs), scheme

    def extent(self, aspect: float) -> tuple[float, float, float, float]:
        t = Transformer.from_crs("EPSG:4326", self.frame.crs, always_xy=True)
        f = self.frame
        xs, ys = zip(*[t.transform(x, y) for x in np.linspace(f.west, f.east, 20) for y in np.linspace(f.south, f.north, 20)])
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (minx + maxx) / 2, (miny + maxy) / 2, maxx - minx, maxy - miny
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def render(self, dpi: int = 160) -> Figure:
        fig = plt.figure(figsize=self.SIZE, dpi=dpi, facecolor=self.theme.paper)
        mx, my, mw, mh = self.RECTS["map"]
        ax = fig.add_axes(self.RECTS["map"])
        aspect = (mw * self.SIZE[0]) / (mh * self.SIZE[1])
        ext = self.extent(aspect)
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(self.theme.water)
        for s in ax.spines.values():
            s.set_edgecolor(self.theme.ink)
            s.set_linewidth(1.3)
        zips, counties, scheme = self._prepare()
        t = Transformer.from_crs("EPSG:4326", self.frame.crs, always_xy=True)
        plt.rcParams["font.family"] = self.theme.font()
        ctx = RenderContext(fig, ax, self.theme, self.metric, scheme, zips, counties, self.ds, self.frame, ext,
                            self.ds.window_label(self.period_end), self.RECTS, self.theme.font(), t.transform)
        for el in self.elements:
            el.draw(ctx)
        ax.set_xlim(ext[0], ext[2])
        ax.set_ylim(ext[1], ext[3])
        ax.set_xlabel("")
        ax.set_ylabel("")
        return fig

    def save(self, path: Path, dpi: int = 160) -> Path:
        fig = self.render(dpi)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor=self.theme.paper)
        plt.close(fig)
        return path
