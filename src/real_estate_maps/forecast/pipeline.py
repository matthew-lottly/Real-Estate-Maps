"""End-to-end run: build features, backtest everything, fit the final model and write forecast files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from .backtest import Band, Metrics, WalkForward, make_model, usable
from .features import FINAL_SET, GROUPS, HORIZONS, SETS, FeatureBuilder

SMOOTH = 3                 # months averaged in the log-price level (damps small-ZIP median noise)
INVESTABLE = 50            # homes sold in the 3-month window; below this a ZIP is too thin to trade on a median


class ForecastPipeline:
    def __init__(self, paths: Paths | None = None):
        self.p = paths or Paths()
        self.out = self.p.processed / "forecast"

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        fb = FeatureBuilder(paths=self.p)
        X, Y, blocks = fb.build(smooth=SMOOTH)
        sold = fb.panel.wide("homes_sold").stack(future_stack=True).reindex(X.index)
        blk = pd.Series(blocks, index=fb.panel.zips)
        wf = WalkForward(X, Y, sold, blocks=blk)
        res: dict = {"smooth_months": SMOOTH, "investable_min_sales": INVESTABLE, "final_set": FINAL_SET, "horizons": {}}
        rules = {"trailing 12-mo growth (momentum)": X["g12_rel"], "neighbors' trailing growth": X["nb_g12"], "cheaper than neighbors": X["nb_gap"],
                 "cheaper than metro": -X["rel_level"], "above own 12-mo trend": X["dev_trend"]}
        preds_all = []
        scales: dict[int, float] = {}
        for h in HORIZONS:
            r: dict = {"years": wf.years(h), "baselines": {}, "models": {}}
            for name, rule in rules.items():
                r["baselines"][name] = {"all": wf.score_rule(rule, h, 0).summary(), "investable": wf.score_rule(rule, h, INVESTABLE).summary()}
            for sn, cols in SETS.items():
                pr = wf.predict(h, cols)
                r["models"][sn] = {"all": Metrics.of(pr["pred"], Y[f"rel_{h}"], sold, 0).summary(), "investable": Metrics.of(pr["pred"], Y[f"rel_{h}"], sold, INVESTABLE).summary()}
                print(h, sn, round(r["models"][sn]["investable"]["rho"], 3), flush=True)
            fin = wf.predict(h, SETS[FINAL_SET], with_interval=True)
            y = Y[f"rel_{h}"].reindex(fin.index)
            d = fin.assign(y=y, s=sold.reindex(fin.index)).dropna()
            mid, hw = (d.hi + d.lo) / 2, (d.hi - d.lo) / 2
            inside = lambda k: ((d.y - mid).abs() <= k * hw)
            ok_rows = d.s >= INVESTABLE
            scale = next((k for k in np.arange(0.5, 4.01, 0.05) if inside(k)[ok_rows].mean() >= 0.8), 4.0)
            scales[h] = float(scale)
            r["interval_scale"] = float(scale)
            r["interval_coverage_80_raw"] = {"all": float(inside(1.0).mean()), "investable": float(inside(1.0)[ok_rows].mean())}
            d["lo"], d["hi"] = mid - scale * hw, mid + scale * hw
            fin["lo"], fin["hi"] = d["lo"].reindex(fin.index), d["hi"].reindex(fin.index)
            r["interval_coverage_80"] = {"all": float(((d.y >= d.lo) & (d.y <= d.hi)).mean()), "investable": float(((d.y >= d.lo) & (d.y <= d.hi))[d.s >= INVESTABLE].mean()),
                                         "mean_width": float((d.hi - d.lo).mean())}
            d["decile"] = d.groupby(level="month")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
            inv = d[d.s >= INVESTABLE]
            inv_dec = inv.groupby(level="month")["pred"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
            r["decile_realized_rel_growth"] = {"all": {int(k): float(v) for k, v in d.groupby("decile")["y"].mean().items()},
                                               "investable": {int(k): float(v) for k, v in inv.groupby(inv_dec)["y"].mean().items()}}
            r["spatial_holdout"] = wf.spatial_holdout(h, SETS[FINAL_SET], INVESTABLE).round(4).to_dict("records")
            gcols = {g: c for g, c in GROUPS.items() if g != "static_context"}
            r["permutation_importance"] = {k: float(v) for k, v in wf.permutation_importance(h, gcols, SETS[FINAL_SET], liquid=INVESTABLE).items()}
            res["horizons"][str(h)] = r
            fin["horizon"] = h
            fin["actual_rel"] = y
            fin["sold"] = sold.reindex(fin.index)
            preds_all.append(fin.reset_index())
        pd.concat(preds_all).to_csv(self.out / "oos_predictions.csv", index=False)
        res["forecast_origin"] = str(wf.last_origin)
        self._final(fb, X, Y, sold, res, scales)
        (self.out / "backtest_summary.json").write_text(json.dumps(res, indent=1, default=float))
        return res

    def _final(self, fb: FeatureBuilder, X: pd.DataFrame, Y: pd.DataFrame, sold: pd.Series, res: dict, scales: dict) -> None:
        """Train on every origin whose outcome is known and predict from the latest month."""
        origin = pd.Period("2026-05", "M")
        months = X.index.get_level_values("month")
        cols = SETS[FINAL_SET]
        now = X[months == origin].copy()
        rows = pd.DataFrame(index=now.index.get_level_values("zip"))
        for h in HORIZONS:
            y = Y[f"rel_{h}"]
            tr = (months <= origin - h) & y.notna().to_numpy()
            c = usable(X, cols, tr)
            lo, hi = y[tr].quantile([.01, .99])
            ytr = y[tr].clip(lo, hi)
            m = make_model().fit(X.loc[tr, c], ytr)
            band = Band(X, ytr, tr, cols)
            rows[f"rel_{h}"] = m.predict(now[c])
            lo_, hi_ = band.predict(now)
            mid, hw = (hi_ + lo_) / 2, (hi_ - lo_) / 2 * scales[h]       # width rescaled so the backtest band covers 80% of outcomes
            rows[f"lo_{h}"], rows[f"hi_{h}"] = mid - hw, mid + hw
        P = fb.panel
        L = P.logp.loc[origin]
        rows["sold_3mo"] = P.wide("homes_sold").loc[origin].reindex(rows.index)
        rows["ppsf"] = P.wide("median_ppsf").loc[origin].reindex(rows.index)
        rows["median_price"] = P.wide("median_sale_price").loc[origin].reindex(rows.index)
        rows["g12_trailing"] = X.loc[months == origin, "g12"].droplevel("month")
        rows["nb_gap"] = X.loc[months == origin, "nb_gap"].droplevel("month")
        rows["months_supply"] = X.loc[months == origin, "months_supply"].droplevel("month")
        rows["dom"] = P.wide("median_dom").loc[origin].reindex(rows.index)
        Ls = P.logp
        hist = Ls.loc[pd.Period("2022-01", "M"):origin]
        rows["drawdown_since_2022"] = (hist.min() - hist.max()).reindex(rows.index)
        rows = rows.join(fb.static[["elev_mean_m", "elev_p10_m", "poi_density", "road_arterial_km_km2", "dist_freeway_km", "dist_downtown_km", "area_km2"]])
        rows["investable"] = rows["sold_3mo"] >= INVESTABLE
        gm = Ls.median(axis=1).dropna()
        res["metro_ppsf_history"] = {"cagr_full": float((np.exp(gm.iloc[-1] - gm.iloc[0]) ** (12 / (len(gm) - 1)) - 1)),
                                     "growth_last_36m": float(np.exp(gm.iloc[-1] - gm.iloc[-37]) - 1), "growth_last_12m": float(np.exp(gm.iloc[-1] - gm.iloc[-13]) - 1),
                                     "first_month": str(gm.index[0]), "last_month": str(gm.index[-1])}
        rows.round(5).to_csv(self.out / "zip_forecast.csv", index_label="zip")
