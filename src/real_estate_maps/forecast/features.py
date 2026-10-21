"""Features known at each origin month, and the forward-growth targets. Nothing here looks past the origin month."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from .geo_features import StaticFeatures
from .panel import Panel

HORIZONS = (12, 24)

GROUPS = {
    "momentum": ["g3", "g6", "g12", "g24", "g36", "accel", "g12_rel", "g24_rel", "dev_trend", "vol12", "rel_level"],
    "market_tightness": ["dom", "dom_yoy", "above_list", "sale_to_list", "months_supply", "newlist_ratio", "sales_yoy", "log_sales"],
    "neighbors": ["nb_g12", "nb_g24", "nb_gap", "nb_hi_g12", "nb_dom", "nb_n"],
    "listings_rdc": ["list_price_yy", "active_yy", "rdc_dom", "rdc_dom_yy", "newlist_yy", "reduced_share", "increased_share", "pending_ratio", "list_ppsf_yy"],
    "demand_hotness": ["hot_score", "hot_demand", "hot_supply", "page_views_vs_us", "page_views_yy"],
    "static_context": ["poi_density", "poi_shopping_density", "poi_food_density", "poi_health_density", "poi_services_density", "poi_education_density", "poi_recreation_density",
                      "poi_culture_density", "poi_branded_share", "poi_diversity", "office_density", "road_arterial_km_km2", "road_residential_km_km2", "road_footway_km_km2",
                      "dist_freeway_km", "dist_downtown_km", "elev_mean_m", "elev_p10_m", "elev_std_m", "area_km2", "hed_resid"],
}
CORE = GROUPS["momentum"] + GROUPS["market_tightness"] + GROUPS["neighbors"]
SETS = {"A_trailing_growth_only": ["g12"],
        "B_sales_market_neighbors": CORE,
        "C_plus_listings": CORE + GROUPS["listings_rdc"],
        "D_plus_hotness": CORE + GROUPS["listings_rdc"] + GROUPS["demand_hotness"],
        "E_plus_static_context": sum(GROUPS.values(), [])}
FINAL_SET = "B_sales_market_neighbors"   # simplest set that matches the richer ones out of time (see backtest ablation)


def neighbor_mean(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Mean of X over each ZIP's adjacent ZIPs, ignoring NaN. X is months x ZIPs, A is a 0/1 adjacency matrix."""
    M = (~np.isnan(X)).astype(float)
    num = np.nan_to_num(X) @ A.T
    den = M @ A.T
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


class FeatureBuilder:
    def __init__(self, panel: Panel | None = None, paths: Paths | None = None):
        self.p = paths or Paths()
        self.panel = panel or Panel(paths=self.p)
        self.static = pd.read_csv(self.p.processed / "houston_zip_static.csv", dtype={"zip": str}).set_index("zip").reindex(self.panel.zips)
        nb = StaticFeatures(self.p).neighbors()
        z = self.panel.zips
        self.A = np.array([[1.0 if b in set(nb.get(a, [])) else 0.0 for b in z] for a in z])

    def _lag(self, X: pd.DataFrame, k: int) -> pd.DataFrame:
        return X.shift(k)

    def hedonic_residual(self, L: pd.DataFrame, blocks: np.ndarray) -> pd.DataFrame:
        """Cross-sectional 'cheap vs fundamentals' signal: each ZIP's log price minus what a ridge fit on *other geographic blocks* predicts from static context."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        cols = [c for c in GROUPS["static_context"] if c != "hed_resid"]
        S = self.static[cols].copy()
        for c in S.columns:
            if c.endswith("density") or c.endswith("km2") or c.startswith("dist") or c == "area_km2":
                S[c] = np.log1p(S[c].clip(lower=0))
        S = S.fillna(S.median())
        Xs = StandardScaler().fit_transform(S.to_numpy())
        out = pd.DataFrame(np.nan, index=L.index, columns=L.columns)
        for t in L.index:
            y = L.loc[t].to_numpy()
            ok = ~np.isnan(y)
            if ok.sum() < 60:
                continue
            pred = np.full(len(y), np.nan)
            for b in np.unique(blocks):
                tr, te = ok & (blocks != b), ok & (blocks == b)
                if tr.sum() > 30 and te.sum():
                    pred[te] = Ridge(alpha=10.0).fit(Xs[tr], y[tr]).predict(Xs[te])
            out.loc[t] = y - pred
        return out

    def blocks(self, k: int = 6) -> np.ndarray:
        from sklearn.cluster import KMeans

        import geopandas as gpd
        z = gpd.read_file(self.p.zip_geojson).set_index("zip").reindex(self.panel.zips)
        c = z.geometry.centroid
        return KMeans(k, n_init=10, random_state=0).fit_predict(np.c_[c.x, c.y])

    def build(self, smooth: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """`smooth` > 1 averages log price over that many consecutive months, which damps the sampling noise of small-ZIP medians."""
        P = self.panel
        L = P.logp
        if smooth > 1:
            L = L.rolling(smooth, min_periods=smooth - 1).mean().where(P.valid)
        zips, months = P.zips, P.months
        blocks = self.blocks()
        F: dict[str, pd.DataFrame] = {}
        for k in (3, 6, 12, 24, 36):
            F[f"g{k}"] = L - self._lag(L, k)
        F["accel"] = F["g6"] - (F["g12"] - F["g6"])
        metro = L.median(axis=1)
        F["rel_level"] = L.sub(metro, axis=0)
        F["g12_rel"] = F["g12"].sub(F["g12"].median(axis=1), axis=0)
        F["g24_rel"] = F["g24"].sub(F["g24"].median(axis=1), axis=0)
        roll = pd.concat([L.shift(i) for i in range(1, 13)]).groupby(level=0).mean() if False else sum(L.shift(i) for i in range(1, 13)) / 12
        F["dev_trend"] = L - roll
        F["vol12"] = L.diff().rolling(12, min_periods=8).std()
        dom, sold, inv, newl = (P.wide(c) for c in ("median_dom", "homes_sold", "inventory", "new_listings"))
        F["dom"] = np.log(dom.where(P.valid))
        F["dom_yoy"] = F["dom"] - F["dom"].shift(12)
        F["above_list"] = P.wide("sold_above_list").where(P.valid)
        F["sale_to_list"] = P.wide("sale_to_list").where(P.valid)
        F["months_supply"] = (inv / (sold / 3)).where(P.valid).clip(upper=36)
        F["newlist_ratio"] = (newl / inv).where(P.valid)
        F["sales_yoy"] = np.log(sold.where(P.valid)) - np.log(sold.where(P.valid)).shift(12)
        F["log_sales"] = np.log(sold.where(P.valid))
        # neighbors (spillover / catch-up)
        for name, src in (("nb_g12", F["g12"]), ("nb_g24", F["g24"]), ("nb_dom", F["dom_yoy"])):
            F[name] = pd.DataFrame(neighbor_mean(src.to_numpy(), self.A), index=months, columns=zips)
        F["nb_gap"] = pd.DataFrame(neighbor_mean(F["rel_level"].to_numpy(), self.A), index=months, columns=zips) - F["rel_level"]
        F["nb_n"] = pd.DataFrame(np.tile(self.A.sum(axis=1), (len(months), 1)), index=months, columns=zips)
        hi = np.full(L.shape, np.nan)
        Lv, G = L.to_numpy(), F["g12"].to_numpy()
        for i in range(len(months)):
            richer = (Lv[i][None, :] > Lv[i][:, None]) & (self.A > 0) & ~np.isnan(G[i])[None, :]
            n = richer.sum(axis=1)
            hi[i] = np.where(n > 0, (richer * np.nan_to_num(G[i])[None, :]).sum(axis=1) / np.where(n > 0, n, 1), np.nan)
        F["nb_hi_g12"] = pd.DataFrame(hi, index=months, columns=zips)
        F.update(P.rdc())
        F["hed_resid"] = self.hedonic_residual(L, blocks)
        # long form
        X = pd.concat({k: v.stack(future_stack=True) for k, v in F.items() if k in sum(GROUPS.values(), [])}, axis=1)
        X.index.names = ["month", "zip"]
        for c in GROUPS["static_context"]:
            if c in self.static.columns and c != "hed_resid":
                X[c] = self.static[c].reindex(X.index.get_level_values("zip")).to_numpy()
        # targets: forward growth, absolute and relative to the cross-section median
        Y = {}
        for h in HORIZONS:
            g = L.shift(-h) - L
            Y[f"abs_{h}"] = g.stack(future_stack=True)
            Y[f"rel_{h}"] = g.sub(g.median(axis=1), axis=0).stack(future_stack=True)
        Y = pd.DataFrame(Y)
        Y.index.names = ["month", "zip"]
        X = X[~L.stack(future_stack=True).reindex(X.index).isna()]           # origins need a valid price today
        return X, Y.reindex(X.index), blocks
