"""Real street network from Overture Maps transportation segments, styled by road class and clipped to the map view."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pyarrow.parquet as pq
import shapely
from shapely.geometry import box

from .inputs import CRS
from .page import halo

ORDER = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service"]
WIDTH = {"motorway": 3.0, "trunk": 2.6, "primary": 2.1, "secondary": 1.7, "tertiary": 1.3, "unclassified": 0.9, "residential": 0.9, "living_street": 0.7, "service": 0.45}
MIN_CLASS_BY_WIDTH_M = [(60_000, 3), (15_000, 4), (9_000, 7), (0, 8)]         # map width -> how many classes of ORDER to draw


class OvertureStreets:
    _cache: gpd.GeoDataFrame | None = None

    def __init__(self, raw_dir):
        self.path = raw_dir / "overture" / "roads_houston.parquet"

    def frame(self) -> gpd.GeoDataFrame:
        if OvertureStreets._cache is None:
            t = pq.read_table(self.path, columns=["names", "class", "geometry"]).to_pandas()
            t = t[t["class"].isin(ORDER)].reset_index(drop=True)
            g = gpd.GeoDataFrame({"cls": t["class"], "name": t["names"].map(lambda n: n["primary"] if n is not None else None)},
                                 geometry=shapely.from_wkb(t["geometry"].to_numpy()), crs="EPSG:4326").to_crs(CRS)
            g.sindex
            OvertureStreets._cache = g
        return OvertureStreets._cache

    def draw(self, page, ax, brand, label: bool = True) -> None:
        x0, y0, x1, y1 = page.extent(ax)
        W = x1 - x0
        n_classes = next(n for w, n in MIN_CLASS_BY_WIDTH_M if W >= w)
        g = self.frame()
        view = box(x0, y0, x1, y1)
        sub = g.iloc[g.sindex.query(view, predicate="intersects")]
        sub = sub[sub["cls"].isin(ORDER[:n_classes])]
        scale = 1.0 if W > 8000 else 1.35
        with page.layer("streets", ax):
            for cls in reversed(ORDER[:n_classes]):
                s = sub[sub["cls"] == cls]
                if s.empty:
                    continue
                lw = WIDTH[cls] * scale
                major = cls in ("motorway", "trunk")
                s.plot(ax=ax, color="#D8D1C2" if not major else "white", lw=lw + 0.7, zorder=3 + 0.01 * ORDER.index(cls), capstyle="round")
                s.plot(ax=ax, color=brand.primary if cls == "motorway" else brand.secondary if cls in ("trunk", "primary") else "white", lw=lw,
                       zorder=3.005 + 0.01 * ORDER.index(cls), capstyle="round")
        if label and n_classes >= 4:
            placed: list[tuple[float, float]] = []
            named = sub[sub["name"].notna() & sub["cls"].isin(ORDER[: min(n_classes, 5)])]
            with page.layer("street-names", ax):
                for name, grp in named.groupby("name"):
                    best = None
                    for geom in grp.geometry:
                        part = geom.intersection(view)
                        if part.is_empty or part.geom_type != "LineString":
                            continue
                        if best is None or part.length > best.length:
                            best = part
                    if best is None or best.length < W * 0.07:
                        continue
                    p = best.interpolate(0.5, normalized=True)
                    if any(np.hypot(p.x - q[0], p.y - q[1]) < W * 0.12 for q in placed):
                        continue
                    a = best.interpolate(0.45, normalized=True)
                    c = best.interpolate(0.55, normalized=True)
                    ang = np.degrees(np.arctan2(c.y - a.y, c.x - a.x))
                    ang = ang + 180 if ang > 90 or ang < -90 else ang
                    placed.append((p.x, p.y))
                    ax.text(p.x, p.y, name, rotation=ang, rotation_mode="anchor", ha="center", va="center", fontsize=5.6, color="#4A5561", zorder=12, path_effects=halo("white", 2.2))
