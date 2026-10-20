"""Data layer: download -> build -> load.

* `RedfinTracker`   streams Redfin's public ZIP-code market tracker (about 1.5 GB gzip) and keeps only one metro area
* `BoundarySource`  downloads ZIP-code (ZCTA) polygons and county outlines
* `DatasetBuilder`  turns raw downloads into small, committed files under data/processed
* `MarketDataset`   loads the processed files and answers questions (latest snapshot, history, ranking)
"""
from __future__ import annotations

import csv
import io
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import pandas as pd
import requests

from .config import Paths


class RedfinTracker:
    """Redfin Data Center: ZIP-code market tracker (rolling 3-month windows, monthly steps). Public data, credit Redfin."""

    URL = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/zip_code_market_tracker.tsv000.gz"
    CHUNK = 1 << 20

    def __init__(self, metro: str = "Houston, TX", session: requests.Session | None = None):
        self.metro, self.s = metro, session or requests.Session()

    def _lines(self, max_bytes: int | None = None) -> Iterator[str]:
        d, tail, got = zlib.decompressobj(16 + zlib.MAX_WBITS), "", 0
        with self.s.get(self.URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(self.CHUNK):
                got += len(chunk)
                tail += d.decompress(chunk).decode("utf-8", "replace")
                *lines, tail = tail.split("\n")
                yield from lines
                if max_bytes and got >= max_bytes:
                    return
        if tail:
            yield tail

    def download(self, dest: Path, max_bytes: int | None = None) -> int:
        """Stream, filter to the metro, write a TSV. Returns the number of rows kept. (`max_bytes` is for tests.)"""
        dest.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with open(dest, "w", newline="", encoding="utf-8") as out:
            w = csv.writer(out, delimiter="\t")
            reader = csv.reader(self._lines(max_bytes), delimiter="\t", quotechar='"')
            header = next(reader)
            w.writerow(header)
            col = header.index("PARENT_METRO_REGION")
            for row in reader:
                if len(row) > col and self.metro in row[col]:
                    w.writerow(row)
                    kept += 1
        return kept


class BoundarySource:
    ZCTA_URL = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/tx_texas_zip_codes_geo.min.json"
    COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

    def __init__(self, session: requests.Session | None = None):
        self.s = session or requests.Session()

    def fetch(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.s.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        return dest

    def download_all(self, raw: Path) -> tuple[Path, Path]:
        return self.fetch(self.ZCTA_URL, raw / "tx_zcta.json"), self.fetch(self.COUNTY_URL, raw / "us_counties.json")


@dataclass
class BuildReport:
    rows: int
    zips: int
    zips_with_polygon: int
    periods: int
    latest_end: str
    missing_polygons: list[str]


class DatasetBuilder:
    """Raw downloads -> small processed files that are committed to the repo."""

    KEEP = {"PERIOD_BEGIN": "period_begin", "PERIOD_END": "period_end", "MEDIAN_SALE_PRICE": "median_sale_price",
            "MEDIAN_SALE_PRICE_YOY": "price_yoy", "MEDIAN_PPSF": "median_ppsf", "MEDIAN_DOM": "median_dom", "HOMES_SOLD": "homes_sold",
            "SOLD_ABOVE_LIST": "sold_above_list", "AVG_SALE_TO_LIST": "sale_to_list", "NEW_LISTINGS": "new_listings", "INVENTORY": "inventory",
            "LAST_UPDATED": "last_updated"}

    def __init__(self, paths: Paths | None = None, simplify_deg: float = 0.0004):
        self.p, self.tol = paths or Paths(), simplify_deg

    def market(self, tsv: Path) -> pd.DataFrame:
        d = pd.read_csv(tsv, sep="\t")
        d = d[d["PROPERTY_TYPE"] == "All Residential"].copy()
        d["zip"] = d["REGION"].str.extract(r"(\d{5})")[0]
        d = d[["zip"] + list(self.KEEP)].rename(columns=self.KEEP)
        d["last_updated"] = d["last_updated"].astype(str).str[:10]
        return d.sort_values(["zip", "period_end"]).reset_index(drop=True)

    def zip_boundaries(self, zcta: Path, zips: set[str]) -> gpd.GeoDataFrame:
        g = gpd.read_file(zcta)[["ZCTA5CE10", "geometry"]].rename(columns={"ZCTA5CE10": "zip"})
        g = g[g["zip"].isin(zips)].copy()
        g["geometry"] = g.geometry.simplify(self.tol, preserve_topology=True)
        return g.set_crs("EPSG:4326", allow_override=True).reset_index(drop=True)

    def counties(self, counties_json: Path) -> gpd.GeoDataFrame:
        g = gpd.read_file(counties_json)
        g = g[g["STATE"] == "48"][["COUNTY", "NAME", "geometry"]].copy()
        g["geometry"] = g.geometry.simplify(self.tol * 4, preserve_topology=True)
        return g.set_crs("EPSG:4326", allow_override=True)

    def build(self, tsv: Path, zcta: Path, counties_json: Path) -> BuildReport:
        m = self.market(tsv)
        zb = self.zip_boundaries(zcta, set(m["zip"]))
        self.p.processed.mkdir(parents=True, exist_ok=True)
        m.to_csv(self.p.market_csv, index=False)
        zb.to_file(self.p.zip_geojson, driver="GeoJSON")
        self.counties(counties_json).to_file(self.p.counties_geojson, driver="GeoJSON")
        missing = sorted(set(m["zip"]) - set(zb["zip"]))
        return BuildReport(len(m), m["zip"].nunique(), zb["zip"].nunique(), m["period_end"].nunique(), m["period_end"].max(), missing)


class MarketDataset:
    """Loads data/processed and serves snapshots as GeoDataFrames."""

    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()
        self.market = pd.read_csv(self.p.market_csv, dtype={"zip": str})
        self.zips = gpd.read_file(self.p.zip_geojson)
        self.counties = gpd.read_file(self.p.counties_geojson)

    def periods(self) -> list[str]:
        return sorted(self.market["period_end"].unique())

    @property
    def latest(self) -> str:
        return self.periods()[-1]

    def snapshot(self, period_end: str | None = None) -> gpd.GeoDataFrame:
        """One row per ZIP polygon for a 3-month window; ZIPs without sales that window keep their geometry with NaN values."""
        end = period_end or self.latest
        rows = self.market[self.market["period_end"] == end]
        out = self.zips.merge(rows, on="zip", how="left")
        out["period_end"] = end
        return out

    def window_label(self, period_end: str | None = None) -> str:
        """'Mar - May 2026' style label for the rolling 3-month window."""
        end = pd.Timestamp(period_end or self.latest)
        start = end - pd.DateOffset(months=2)
        same_year = start.year == end.year
        return f"{start:%b} – {end:%b %Y}" if same_year else f"{start:%b %Y} – {end:%b %Y}"

    def last_updated(self) -> str:
        return str(self.market["last_updated"].max())

    def history(self, zip_code: str) -> pd.DataFrame:
        return self.market[self.market["zip"] == zip_code].sort_values("period_end")
