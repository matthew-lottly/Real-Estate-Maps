"""ZIP x month panel: Redfin sales metrics joined with Realtor.com listing and demand metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths

MIN_SALES = 10        # a ZIP-month only counts if the 3-month window has at least this many sales


class Panel:
    """Wide matrices (months x ZIPs) for each series. A month is the 3-month window *ending* that month."""

    RDC_CORE = {"median_listing_price": "list_price", "median_listing_price_yy": "list_price_yy", "active_listing_count_yy": "active_yy",
                "median_days_on_market": "rdc_dom", "median_days_on_market_yy": "rdc_dom_yy", "new_listing_count_yy": "newlist_yy",
                "price_reduced_share": "reduced_share", "price_increased_share": "increased_share", "pending_ratio": "pending_ratio",
                "median_listing_price_per_square_foot_yy": "list_ppsf_yy"}
    RDC_HOT = {"hotness_score": "hot_score", "demand_score": "hot_demand", "supply_score": "hot_supply", "page_view_count_per_property_vs_us": "page_views_vs_us",
               "page_view_count_per_property_yy": "page_views_yy"}

    def __init__(self, paths: Paths | None = None, zips: list[str] | None = None):
        self.p = paths or Paths()
        m = pd.read_csv(self.p.market_csv, dtype={"zip": str})
        m["month"] = pd.to_datetime(m["period_end"]).dt.to_period("M")
        self.long = m
        poly = set(pd.read_csv(self.p.processed / "houston_zip_static.csv", dtype={"zip": str})["zip"]) if (self.p.processed / "houston_zip_static.csv").exists() else set(m["zip"])
        self.zips = sorted(zips or (set(m["zip"]) & poly))
        self.months = pd.period_range(m["month"].min(), m["month"].max(), freq="M")
        self._w: dict[str, pd.DataFrame] = {}

    def wide(self, col: str) -> pd.DataFrame:
        if col not in self._w:
            w = self.long.pivot_table(index="month", columns="zip", values=col, aggfunc="last")
            self._w[col] = w.reindex(index=self.months, columns=self.zips)
        return self._w[col]

    @property
    def valid(self) -> pd.DataFrame:
        return (self.wide("homes_sold") >= MIN_SALES) & self.wide("median_ppsf").notna()

    @property
    def logp(self) -> pd.DataFrame:
        return np.log(self.wide("median_ppsf").where(self.valid))

    def rdc(self) -> dict[str, pd.DataFrame]:
        out = {}
        for key, cols in (("core", self.RDC_CORE), ("hot", self.RDC_HOT)):
            d = pd.read_csv(self.p.raw / "rdc" / f"houston_{key}.csv", dtype={"postal_code": str})
            d["month"] = pd.to_datetime(d["month_date_yyyymm"].astype(str), format="%Y%m").dt.to_period("M")
            for src, name in cols.items():
                w = d.pivot_table(index="month", columns="postal_code", values=src, aggfunc="last")
                out[name] = w.reindex(index=self.months, columns=self.zips)
        return out
