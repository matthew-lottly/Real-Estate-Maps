"""Walk-forward backtest, ablations, spatial hold-out and conformal intervals.

Every model is trained only on origins whose outcome was already known before the test year began (a purge of `h` months), so no test
outcome leaks into training. Skill is measured *within* each origin month (how well the model ranks ZIPs against each other), because the
question for an investor is which ZIPs to pick, not what the whole metro will do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

PARAMS = dict(max_depth=4, learning_rate=0.04, max_iter=250, min_samples_leaf=80, l2_regularization=2.0, random_state=0)


def make_model(q: float | None = None) -> HGB:
    return HGB(loss="quantile", quantile=q, **PARAMS) if q else HGB(**PARAMS)


def usable(X: pd.DataFrame, cols: list[str], mask) -> list[str]:
    """Columns with enough real values in the training rows (a feature that did not exist yet cannot be learned)."""
    return [c for c in cols if X.loc[mask, c].notna().sum() > 200 and X.loc[mask, c].nunique() > 3]


class Band:
    """Conformalised 10-90% band (split conformal quantile regression): the last year of training origins calibrates the width so that
    roughly 80% of outcomes should land inside."""

    def __init__(self, X: pd.DataFrame, ytr: pd.Series, tr, cols: list[str]):
        months = X.index.get_level_values("month")
        trm = np.asarray(tr)
        cal = trm & (months > months[trm].max() - 12)
        fit = trm & ~cal
        self.cols = usable(X, cols, fit)
        yfit = ytr.reindex(X.index[fit])
        self.lo_m, self.hi_m = make_model(0.1).fit(X.loc[fit, self.cols], yfit), make_model(0.9).fit(X.loc[fit, self.cols], yfit)
        yc = ytr.reindex(X.index[cal]).to_numpy()
        xc = X.loc[cal, self.cols]
        score = np.maximum(self.lo_m.predict(xc) - yc, yc - self.hi_m.predict(xc))
        self.adj = float(np.quantile(score, min(0.8 * (1 + 1 / len(score)), 1.0)))

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.lo_m.predict(X[self.cols]) - self.adj, self.hi_m.predict(X[self.cols]) + self.adj


@dataclass
class Metrics:
    rows: pd.DataFrame

    @staticmethod
    def of(pred: pd.Series, y: pd.Series, sold: pd.Series, liquid: int = 0, min_n: int = 40) -> "Metrics":
        d = pd.DataFrame({"p": pred, "y": y.reindex(pred.index), "s": sold.reindex(pred.index)}).dropna()
        if liquid:
            d = d[d["s"] >= liquid]
        out = []
        for mo, g in d.groupby(level="month"):
            if len(g) < min_n:
                continue
            q, dec = max(len(g) // 5, 1), max(len(g) // 10, 1)
            top = g.sort_values("p", ascending=False)
            act = set(g.sort_values("y", ascending=False).head(q).index)
            out.append({"year": mo.year, "rho": spearmanr(g["p"], g["y"])[0], "precision20": len(set(top.head(q).index) & act) / q,
                        "top10": top.head(dec)["y"].mean(), "bottom10": top.tail(dec)["y"].mean(), "n": len(g)})
        return Metrics(pd.DataFrame(out))

    def summary(self) -> dict:
        r = self.rows
        by = r.groupby("year")["rho"].mean()
        return {"rho": float(r["rho"].mean()), "precision20": float(r["precision20"].mean()), "top10": float(r["top10"].mean()), "bottom10": float(r["bottom10"].mean()),
                "spread": float((r["top10"] - r["bottom10"]).mean()), "zips_per_month": float(r["n"].mean()), "origins": int(len(r)),
                "rho_by_year": {int(k): float(v) for k, v in by.items()}, "years_positive": int((by > 0).sum()), "years": int(len(by))}


@dataclass
class WalkForward:
    X: pd.DataFrame
    Y: pd.DataFrame
    sold: pd.Series
    blocks: pd.Series | None = None          # ZIP -> geographic block id
    first_test_year: int = 2020
    last_origin: pd.Period = field(default_factory=lambda: pd.Period("2026-05", "M"))

    def _fold(self, h: int, year: int):
        months = self.X.index.get_level_values("month")
        y = self.Y[f"rel_{h}"]
        first = pd.Period(f"{year}-01", "M")
        ok = y.notna().to_numpy()
        tr = (months <= first - h - 1) & ok
        te = (months >= first) & (months <= pd.Period(f"{year}-12", "M")) & ok
        return tr, te

    def years(self, h: int) -> list[int]:
        return [y for y in range(self.first_test_year, self.last_origin.year + 1) if self._fold(h, y)[1].sum() > 0]

    def predict(self, h: int, cols: list[str], with_interval: bool = False) -> pd.DataFrame:
        """Out-of-time predictions for every test origin. With intervals: conformalised 10-90% quantile band."""
        y = self.Y[f"rel_{h}"]
        parts = []
        for yr in self.years(h):
            tr, te = self._fold(h, yr)
            c = usable(self.X, cols, tr)
            lo, hi = y[tr].quantile([.01, .99])
            ytr = y[tr].clip(lo, hi)
            m = make_model().fit(self.X.loc[tr, c], ytr)
            out = pd.DataFrame({"pred": m.predict(self.X.loc[te, c])}, index=self.X.index[te])
            if with_interval:
                band = Band(self.X, ytr, tr, cols)
                out["lo"], out["hi"] = band.predict(self.X.loc[te])
            parts.append(out)
        return pd.concat(parts)

    def score_rule(self, rule: pd.Series, h: int, liquid: int = 0) -> Metrics:
        te = np.zeros(len(self.X), dtype=bool)
        for yr in self.years(h):
            te |= self._fold(h, yr)[1]
        return Metrics.of(rule[te], self.Y[f"rel_{h}"], self.sold, liquid)

    def spatial_holdout(self, h: int, cols: list[str], liquid: int = 0) -> pd.DataFrame:
        """Train on ZIPs from the other geographic blocks (and earlier origins), test on the held-out block: does it work on ZIPs the model never saw?"""
        y = self.Y[f"rel_{h}"]
        zips = self.X.index.get_level_values("zip")
        blk = zips.map(self.blocks).to_numpy()
        rows = []
        for b in sorted(set(blk[~pd.isna(blk)])):
            parts = []
            for yr in self.years(h):
                tr, te = self._fold(h, yr)
                trb, teb = tr & (blk != b), te & (blk == b)
                if trb.sum() < 500 or teb.sum() == 0:
                    continue
                c = usable(self.X, cols, trb)
                lo, hi = y[trb].quantile([.01, .99])
                m = make_model().fit(self.X.loc[trb, c], y[trb].clip(lo, hi))
                parts.append(pd.Series(m.predict(self.X.loc[teb, c]), index=self.X.index[teb]))
            if parts:
                m_ = Metrics.of(pd.concat(parts), y, self.sold, liquid, min_n=6)
                if m_.rows.empty:
                    continue
                s = m_.summary()
                rows.append({"block": int(b), "rho": s["rho"], "precision20": s["precision20"], "top10": s["top10"], "zips_in_block": int(len(set(zips[blk == b])))})
        return pd.DataFrame(rows)

    def permutation_importance(self, h: int, groups: dict[str, list[str]], cols: list[str], repeats: int = 3, liquid: int = 0) -> pd.Series:
        """Drop in cross-sectional rank correlation when a feature group is shuffled across rows of the test set."""
        rng = np.random.default_rng(0)
        y = self.Y[f"rel_{h}"]
        tot = {g: [] for g in groups}
        base_all = []
        for yr in self.years(h):
            tr, te = self._fold(h, yr)
            c = usable(self.X, cols, tr)
            lo, hi = y[tr].quantile([.01, .99])
            m = make_model().fit(self.X.loc[tr, c], y[tr].clip(lo, hi))
            Xte = self.X.loc[te, c]
            base = Metrics.of(pd.Series(m.predict(Xte), index=Xte.index), y, self.sold, liquid).summary()["rho"]
            base_all.append(base)
            for g, gc in groups.items():
                gc = [k for k in gc if k in c]
                if not gc:
                    continue
                drops = []
                for _ in range(repeats):
                    Xp = Xte.copy()
                    idx = rng.permutation(len(Xp))
                    Xp[gc] = Xte[gc].to_numpy()[idx]
                    drops.append(base - Metrics.of(pd.Series(m.predict(Xp), index=Xp.index), y, self.sold, liquid).summary()["rho"])
                tot[g].append(np.mean(drops))
        return pd.Series({g: float(np.mean(v)) for g, v in tot.items() if v}).sort_values(ascending=False)
