"""ZIP-level static context from Overture POIs and roads, USGS elevation, and ZIP adjacency.

These describe slow-moving structure (how amenity-rich, connected and high-lying a ZIP is). They come from a single recent
snapshot, so in a backtest they carry some look-ahead (today's amenities partly reflect past growth). The backtest reports a
with/without ablation so that effect is visible instead of hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import rasterio
from rasterio.mask import mask as rio_mask
from shapely import wkb
from shapely.geometry import box, mapping

from ..config import Paths
from .sources import DEM_TILES

CRS = "EPSG:3081"
DOWNTOWN = (-95.3698, 29.7604)
GROUPS = {"shopping": "shopping", "food": "food_and_drink", "health": "health_care", "services": "services_and_business", "education": "education",
          "recreation": "sports_and_recreation", "culture": "arts_and_entertainment", "lodging": "lodging"}


class StaticFeatures:
    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()
        self.zips = gpd.read_file(self.p.zip_geojson).to_crs(CRS)
        self.zips["area_km2"] = self.zips.geometry.area / 1e6

    # --- places ---
    def places(self) -> pd.DataFrame:
        t = pq.read_table(self.p.raw / "overture" / "places_houston.parquet", columns=["taxonomy", "basic_category", "brand", "confidence", "operating_status", "geometry"]).to_pandas()
        t = t[(t["operating_status"] == "open") & (t["confidence"] >= 0.6)].copy()
        t["grp"] = t["taxonomy"].map(lambda x: x["hierarchy"][0] if x is not None and len(x["hierarchy"]) else None)
        t["branded"] = t["brand"].notna()
        g = gpd.GeoDataFrame(t[["grp", "basic_category", "branded"]], geometry=[wkb.loads(b) for b in t["geometry"]], crs="EPSG:4326").to_crs(CRS)
        return g

    def poi_features(self) -> pd.DataFrame:
        pl = self.places()
        j = gpd.sjoin(pl, self.zips[["zip", "geometry"]], predicate="within")
        out = pd.DataFrame(index=self.zips["zip"])
        area = self.zips.set_index("zip")["area_km2"]
        out["poi_density"] = j.groupby("zip").size().reindex(out.index).fillna(0) / area
        for name, h in GROUPS.items():
            out[f"poi_{name}_density"] = j[j["grp"] == h].groupby("zip").size().reindex(out.index).fillna(0) / area
        out["poi_branded_share"] = j.groupby("zip")["branded"].mean().reindex(out.index)
        cnt = j.groupby(["zip", "grp"]).size().unstack(fill_value=0)
        p = cnt.div(cnt.sum(axis=1), axis=0)
        out["poi_diversity"] = (-(p * np.log(p.where(p > 0, 1))).sum(axis=1)).reindex(out.index)
        office = j[j["basic_category"].isin(["corporate_or_business_office", "professional_service", "technical_service", "b2b_service"])]
        out["office_density"] = office.groupby("zip").size().reindex(out.index).fillna(0) / area
        return out

    # --- roads ---
    def road_features(self) -> pd.DataFrame:
        t = pq.read_table(self.p.raw / "overture" / "roads_houston.parquet", columns=["class", "geometry"]).to_pandas()
        g = gpd.GeoDataFrame(t[["class"]], geometry=[wkb.loads(b) for b in t["geometry"]], crs="EPSG:4326").to_crs(CRS)
        zp = self.zips[["zip", "geometry", "area_km2"]]
        clipped = gpd.overlay(g[g["class"].isin(["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "footway"])], zp, how="intersection", keep_geom_type=True)
        clipped["km"] = clipped.geometry.length / 1000
        piv = clipped.pivot_table(index="zip", columns="class", values="km", aggfunc="sum", fill_value=0)
        area = zp.set_index("zip")["area_km2"]
        out = pd.DataFrame(index=zp["zip"])
        for c in ("motorway", "trunk", "primary", "secondary", "tertiary", "residential", "footway"):
            out[f"road_{c}_km_km2"] = (piv[c] if c in piv else 0) / area
        out["road_arterial_km_km2"] = out[["road_primary_km_km2", "road_secondary_km_km2", "road_trunk_km_km2"]].sum(axis=1)
        fw = g[g["class"].isin(["motorway", "trunk"])].geometry.union_all()
        cen = zp.set_index("zip").geometry.centroid
        out["dist_freeway_km"] = [fw.distance(c) / 1000 for c in cen]
        out["dist_downtown_km"] = [c.distance(gpd.GeoSeries.from_xy([DOWNTOWN[0]], [DOWNTOWN[1]], crs="EPSG:4326").to_crs(CRS).iloc[0]) / 1000 for c in cen]
        return out

    # --- elevation ---
    def elevation_features(self) -> pd.DataFrame:
        srcs = [rasterio.open(self.p.raw / "dem" / f"USGS_1_{t}.tif") for t in DEM_TILES]
        rows = {}
        z4326 = self.zips.to_crs("EPSG:4326")
        for zc, geom in zip(z4326["zip"], z4326.geometry):
            vals = []
            for s in srcs:
                if not box(*s.bounds).intersects(geom):
                    continue
                try:
                    arr, _ = rio_mask(s, [mapping(geom)], crop=True, filled=False)
                except ValueError:
                    continue
                a = arr[0].compressed()
                if a.size:
                    vals.append(a)
            if vals:
                v = np.concatenate(vals)
                rows[zc] = {"elev_mean_m": float(v.mean()), "elev_p10_m": float(np.percentile(v, 10)), "elev_std_m": float(v.std())}
        return pd.DataFrame(rows).T

    # --- adjacency ---
    def neighbors(self) -> dict[str, list[str]]:
        z = self.zips[["zip", "geometry"]].copy()
        z["geometry"] = z.geometry.buffer(60)             # tolerate slivers between simplified polygons
        j = gpd.sjoin(z, z, predicate="intersects")
        j = j[j["zip_left"] != j["zip_right"]]
        return j.groupby("zip_left")["zip_right"].apply(lambda s: sorted(set(s))).to_dict()

    def build(self) -> pd.DataFrame:
        f = self.poi_features().join(self.road_features()).join(self.elevation_features())
        f["area_km2"] = self.zips.set_index("zip")["area_km2"]
        out = self.p.processed / "houston_zip_static.csv"
        f.round(5).to_csv(out, index_label="zip")
        return f
