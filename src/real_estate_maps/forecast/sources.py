"""Public datasets beyond Redfin, all fetched from hosts reachable without credentials.

* Realtor.com Research (RDC) ZIP listing metrics and Hotness (demand) index, monthly, 2016/2017 -> now
* Overture Maps places (POIs) and road segments, read straight from S3 with a bounding-box filter
* USGS 3DEP 1 arc-second elevation tiles (The National Map staged products)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

from ..config import Paths

RDC = "https://econdata.s3-us-west-2.amazonaws.com/Reports"
RDC_FILES = {"core": "Core/RDC_Inventory_Core_Metrics_Zip_History.csv", "hot": "Hotness/RDC_Inventory_Hotness_Metrics_Zip_History.csv"}
OVERTURE_RELEASE = "2026-09-23.1"
BBOX = (-96.2, 29.1, -94.5, 30.5)            # Houston region, WGS84: W, S, E, N
DEM_TILES = ("n30w097", "n30w096", "n30w095", "n31w097", "n31w096", "n31w095")
DEM_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/{t}/USGS_1_{t}.tif"


class ExtraSources:
    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()

    # --- Realtor.com ---
    def fetch_rdc(self) -> dict[str, Path]:
        """Stream each national ZIP file in chunks and keep only the ZIPs in our Houston market file."""
        zips = set(pd.read_csv(self.p.market_csv, dtype={"zip": str})["zip"])
        out = {}
        for key, rel in RDC_FILES.items():
            dest = self.p.raw / "rdc" / f"houston_{key}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                tmp = dest.with_suffix(".download")
                with requests.get(f"{RDC}/{rel}", stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for ch in r.iter_content(1 << 20):
                            fh.write(ch)
                parts = []
                for ch in pd.read_csv(tmp, dtype={"postal_code": str}, chunksize=500_000, low_memory=False):
                    ch["postal_code"] = ch["postal_code"].str.zfill(5)
                    parts.append(ch[ch["postal_code"].isin(zips)])
                pd.concat(parts).to_csv(dest, index=False)
                tmp.unlink()
            out[key] = dest
        return out

    # --- Overture ---
    def fetch_overture(self) -> dict[str, Path]:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        from pyarrow import fs

        s3 = fs.S3FileSystem(anonymous=True, region="us-west-2", proxy_options=os.environ.get("HTTPS_PROXY"))
        W, S, E, N = BBOX
        box = (ds.field("bbox", "xmin") > W) & (ds.field("bbox", "xmax") < E) & (ds.field("bbox", "ymin") > S) & (ds.field("bbox", "ymax") < N)
        R = f"overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
        out = {}
        jobs = {"places": ("theme=places/type=place", box, ["id", "names", "taxonomy", "basic_category", "confidence", "brand", "operating_status", "geometry", "bbox"]),
                "roads": ("theme=transportation/type=segment", box & (ds.field("subtype") == "road"), ["id", "names", "class", "subclass", "road_surface", "speed_limits", "geometry", "bbox"])}
        for key, (path, flt, cols) in jobs.items():
            dest = self.p.raw / "overture" / f"{key}_houston.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                pq.write_table(ds.dataset(f"{R}/{path}", filesystem=s3, format="parquet").to_table(filter=flt, columns=cols), dest)
            out[key] = dest
        return out

    # --- USGS elevation ---
    def fetch_dem(self) -> list[Path]:
        d = self.p.raw / "dem"
        d.mkdir(parents=True, exist_ok=True)
        got = []
        for t in DEM_TILES:
            dest = d / f"USGS_1_{t}.tif"
            if not dest.exists():
                with requests.get(DEM_URL.format(t=t), stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as fh:
                        for ch in r.iter_content(1 << 20):
                            fh.write(ch)
            got.append(dest)
        return got
