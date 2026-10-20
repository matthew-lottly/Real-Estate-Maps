"""Turning numbers into map classes: quantiles, natural breaks (Jenks) or fixed breaks."""
from __future__ import annotations

import abc
from dataclasses import dataclass

import mapclassify
import numpy as np

from .metrics import Metric


@dataclass
class ClassScheme:
    """Ordered classes: `edges` has n+1 values (min..max); `colors` and `labels` have n entries."""

    edges: list[float]
    colors: list[str]
    labels: list[str]

    @property
    def n(self) -> int:
        return len(self.colors)

    def assign(self, values) -> np.ndarray:
        """Class index per value (NaN -> -1)."""
        v = np.asarray(values, dtype=float)
        idx = np.clip(np.searchsorted(self.edges[1:-1], v, side="right"), 0, self.n - 1)
        return np.where(np.isnan(v), -1, idx)

    def counts(self, values) -> list[int]:
        idx = self.assign(values)
        return [int((idx == i).sum()) for i in range(self.n)]


class Classifier(abc.ABC):
    def __init__(self, metric: Metric):
        self.m = metric

    @abc.abstractmethod
    def edges(self, values: np.ndarray) -> list[float]: ...

    def scheme(self, values) -> ClassScheme:
        v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
        edges = self.edges(v)
        n = len(edges) - 1
        pal = self.m.palette
        colors = pal.colors(n) if pal.diverging is False else self._diverging_colors(n)
        labels = [self._label(edges, i) for i in range(n)]
        return ClassScheme(edges, colors, labels)

    def _diverging_colors(self, n: int) -> list[str]:
        return self.m.palette.colors(n)

    def _label(self, e: list[float], i: int) -> str:
        f = self.m.short
        if e[0] < 0:
            if i == 0:
                return f"below {f(e[1])}"
            if i == len(e) - 2:
                return f"above {f(e[i])}"
            return f"{f(e[i])} to {f(e[i + 1])}"
        if i == 0:
            return f"< {f(e[1])}"
        if i == len(e) - 2:
            return f"{f(e[i])}+"
        return f"{f(e[i])}–{f(e[i + 1])}"

    @staticmethod
    def _nice(edges: list[float], v: np.ndarray) -> list[float]:
        e = sorted(set(float(x) for x in edges))
        e[0], e[-1] = float(v.min()), float(v.max())
        return e


class QuantileClassifier(Classifier):
    def edges(self, values):
        q = mapclassify.Quantiles(values, k=self.m.classes)
        return self._nice([values.min(), *q.bins], values)


class NaturalBreaksClassifier(Classifier):
    def edges(self, values):
        nb = mapclassify.NaturalBreaks(values, k=min(self.m.classes, max(len(np.unique(values)), 2)), initial=200)
        return self._nice([values.min(), *nb.bins], values)


class FixedBreaksClassifier(Classifier):
    def edges(self, values):
        return [float(values.min()) if len(values) else 0.0, *self.m.breaks, float(values.max()) if len(values) else 1.0]


class ClassifierFactory:
    @staticmethod
    def for_metric(m: Metric) -> Classifier:
        return {"quantile": QuantileClassifier, "natural": NaturalBreaksClassifier, "fixed": FixedBreaksClassifier}[m.scheme](m)
