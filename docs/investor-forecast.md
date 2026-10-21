# Houston investor forecast: data, method and how far to trust it

Question: **which Houston ZIP codes are likely to outperform the rest of the market over the next 12 and 24 months, and how sure are we?**
Outputs: `investor_forecast_24m`, `investor_forecast_12m`, `investor_risk_return`, `investor_evidence` (letter-landscape PNG/PDF/SVG in `output/commercial/`), plus `data/processed/forecast/zip_forecast.csv` and `backtest_summary.json`.

## Data used (all public, no keys)

| Source | What | Use |
|---|---|---|
| Redfin Data Center ZIP tracker | monthly 3-month windows 2012-03 to 2026-05: median sale price and $/sq ft, days on market, sales, sale-to-list, new listings, inventory | target and core features |
| Realtor.com Research (S3) | ZIP listing metrics 2016+, Hotness/demand index 2017-08 to 2026-02 | tested as features |
| Overture Maps 2026-09-23.1 | 420k places (open, with brand), 885k road segments | amenity density, road access, real street basemap, nearest-amenity lists |
| USGS 3DEP 1 arc-second | elevation | low-lying risk flag |
| Census 2010 ZCTA polygons | ZIP areas, adjacency | neighbor spillover features |

Looked for and **not reachable** from this environment: Census ACS/TIGER, LEHD jobs, IRS income by ZIP, FEMA flood zones, Zillow, FHFA, HUD, building permits, HCAD parcels. So the model has no income, jobs, rents, permits or flood-zone data. Each is a clear next upgrade.

## Method

- **Target:** growth in median sale price per sq ft over the next 12 / 24 months, **relative to the typical ZIP that month** (the metro-wide move is not predictable from ZIP features; picking winners is). Price level is a 3-month average of log $/sq ft to damp small-sample noise.
- **Features known at the forecast date only:** own trend, price vs metro and vs 12-month trend, days on market, months of supply, sales volume, and **spillover from adjacent ZIPs** (their growth, and the price gap to them). Grouped as momentum/level, market tightness, neighbors.
- **Model:** gradient-boosted trees (scikit-learn `HistGradientBoostingRegressor`), plus quantile models for an 80% range, widened until the backtest band covers 80%.
- **Backtest:** walk-forward by year, 2020 onward. Each fold trains only on origins whose outcome was already known before the test year (a purge of the horizon), tested by `tests/test_forecast.py`. Skill is measured **within each month across ZIPs** (rank correlation, top-fifth hit rate, return of the top and bottom decile).
- **Investable universe:** ZIPs with at least 50 sales in the window. Thin ZIPs have noisy medians that flatter the numbers.
- **Checks:** simple-rule baselines, ablation of feature groups, held-out geography (train on other regions), permutation importance, interval coverage.

## What it found (24-month horizon, investable ZIPs, 2020-2024 test years)

- Rank correlation with actual outcomes **+0.22**, positive in 5 of 5 years. Model top-decile ZIPs beat the typical ZIP by **+3.4%** over 24 months, bottom decile **-4.0%**; 30% of its top-fifth picks were really in the top fifth (chance 20%). At 12 months: +0.22, +2.6% / -2.8%, 6 of 6 years.
- **Momentum mean-reverts.** Following recent growth scores -0.08 (positive in 1 of 6 years). ZIPs that ran ahead lag; ZIPs cheap next to richer neighbors catch up. This is the opposite of the "buy what's hot" instinct.
- Neighbor features and market tightness add real skill. **Realtor.com listings added nothing** (+0.20 vs +0.22). Hotness adds about +0.02 but isn't published for the latest months. **Amenity, road and elevation features hurt out of time** (they let the model memorize ZIP identity), so they are used for risk flags and context, not prediction.
- Skill holds on regions held out of training (positive in every region with at least 10 ZIPs).
- Noise matters: all-ZIP rank correlation is +0.28 vs +0.22 for liquid ZIPs. Trust the liquid number.
- Raw 80% ranges covered only 69% of outcomes at 24 months, so they were rescaled by 1.35 to reach 80%.

## Limits, stated plainly

- It is a modest edge, not a forecast of the metro. Houston's median $/sq ft is up only 0.4% over 36 months, so metro direction dominates absolute returns; the maps show outperformance versus the typical ZIP and an illustrative payoff table under three metro scenarios.
- Price appreciation only: no rents, yield, costs, taxes or financing. The model is trained on a few regimes (2020-22 boom, 2022-26 flat); a different regime can change its behavior.
- Only 5-6 independent test years, with overlapping monthly forecasts, so confidence intervals on the skill numbers are wide.
- ZIP areas are Census 2010 ZCTAs; 26 newer ZIPs have no polygon and are not scored. ZIPs are postal areas, not neighborhoods or submarkets.
- Low-elevation flags are a screen, not flood risk. FEMA flood data was not reachable.
- Static-snapshot features (amenities, roads) are measured today; they carry look-ahead in any backtest, which is another reason they stay out of the prediction.

## Reproduce

```bash
python -m real_estate_maps fetch-extra     # Realtor.com, Overture, USGS elevation
python -m real_estate_maps build-static    # ZIP-level amenity / road / elevation features
python -m real_estate_maps forecast        # features, backtest, forecast (a few minutes)
python -m real_estate_maps commercial --map forecast-24   # or risk-return, evidence, forecast-12
```
