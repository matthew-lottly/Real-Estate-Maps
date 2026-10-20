"""Data we can't bundle. Each is an interface with an honest default, so a map never fakes what it doesn't have."""
from __future__ import annotations

import abc
import json
from pathlib import Path

import matplotlib.image as mpimg

from .inputs import to_xy


class DriveTimeProvider(abc.ABC):
    """Drive-time bands need a routing engine (OpenRouteService, Valhalla, ArcGIS, Google). Subclass and return a polygon per minute value."""

    available = False

    @abc.abstractmethod
    def isochrone(self, lon: float, lat: float, minutes: int): ...


class NoDriveTime(DriveTimeProvider):
    """Default: no routing service is reachable from the build environment, so maps show straight-line distance instead and say so."""

    def isochrone(self, lon, lat, minutes):
        raise NotImplementedError("No routing provider configured; maps use straight-line distance.")


class ImageryProvider(abc.ABC):
    available = False

    @abc.abstractmethod
    def draw(self, ax, extent) -> bool: ...


class NoImagery(ImageryProvider):
    """Default: licensed aerial imagery isn't bundled. The parcel map draws a labelled placeholder instead."""

    def draw(self, ax, extent) -> bool:
        return False


class ImageFileImagery(ImageryProvider):
    """Drop a north-up aerial PNG/JPG plus a sidecar `<name>.json` {"west":..,"south":..,"east":..,"north":..} (WGS84) and it's used automatically."""

    available = True

    def __init__(self, image: Path):
        self.image = Path(image)
        self.bounds = json.loads(self.image.with_suffix(".json").read_text())

    def draw(self, ax, extent) -> bool:
        b = self.bounds
        x0, y0 = to_xy(b["west"], b["south"])
        x1, y1 = to_xy(b["east"], b["north"])
        ax.imshow(mpimg.imread(self.image), extent=(x0, x1, y0, y1), zorder=1, interpolation="lanczos")
        return True

    @classmethod
    def find(cls, folder: Path) -> ImageryProvider:
        for ext in (".png", ".jpg"):
            p = Path(folder) / f"aerial{ext}"
            if p.exists() and p.with_suffix(".json").exists():
                return cls(p)
        return NoImagery()
