# Commercial real estate map set (Houston demo)

Built for a marketing studio that produces offering memorandums, brochures and flyers. Every map is a letter-landscape page, **3300 x 2550 px at 300 ppi**, exported as PNG, PDF and an SVG whose groups are named layers (`roads`, `parcel-boundary`, `callouts`, `scale-bar`...).

```bash
python -m real_estate_maps build-context          # one time: highways, cities, airports, water
python -m real_estate_maps commercial             # all nine maps from inputs/sample
python -m real_estate_maps commercial --inputs path/to/client_folder --map rings
```

## Map types

| Key | Map | Real data used | What is gated |
|---|---|---|---|
| `regional` | Regional access | Natural Earth highways/cities/airports/Gulf; straight-line distances | Drive times (no routing service reachable). `DriveTimeProvider` is the plug-in point. Highways are generalized majors only. |
| `amenities` | Amenity map | Client CSV (`name,category,lon,lat,logo`) | Logos: supplied PNGs used, otherwise monogram badge. Sample list: 3 retail centers + 1 park with approximate coordinates; restaurants and employers are labelled samples. No street-level roads. |
| `parcel` | Aerial parcel | Parcel polygon -> acreage and side lengths computed from geometry | Aerial imagery not bundled; placeholder is stamped. Drop `aerial.png` + `aerial.json` (WGS84 bounds) in the inputs folder. Demo parcel is synthetic. |
| `rings` | Demographic rings | 1/3/5 mi rings; Redfin ZIP stats area-apportioned (homes sold, weighted price, sales volume, vs metro) | Population/households/income need a client extract (`demographics.csv`: `ring_miles,metric,value`). None bundled. |
| `neighborhood` | Neighborhood market | Redfin median price by ZIP, labelled | ZIP is the finest level in this data, not neighborhood boundaries. |
| `locator` | Site location (market area + metro + Texas frames) | Natural Earth highways/water, county and ZIP outlines | Generalized roads only; no street basemap. |
| `trends` | Market momentum | Redfin YoY by ZIP + five-year price paths (4-window average) for the 4 nearest ZIPs vs metro | Small-ZIP series are noisy; housing data, not commercial rents. |
| `suitability-retail`, `suitability-industrial` | Suitability screening | Weighted overlay of percentile-ranked ZIP housing factors + distance to highways/airports | Housing-market proxies, not rents, zoning or flood. Weights are editable `Profile`s. |

## Inputs a client supplies
Folder with `brand.json` (colors, font families, logo path), `parcel.geojson` (WGS84), `amenities.csv` (+ `logos/`), optional `demographics.csv`, `aerial.png/json`. Without a parcel file the synthetic sample site is used and every page is stamped SAMPLE.

## Not delivered yet (be honest in a proposal)
- **Layered InDesign / Illustrator / Photoshop files.** The SVG opens in Illustrator, Affinity and Inkscape with named groups; native .ai/.indd/.psd are not produced.
- Matching a client's sample style automatically; brand fonts must be installed locally (fallback: Inter/DejaVu).
- Aerial imagery, drive-time isochrones, street basemap, Census/vendor demographics: need licensed or reachable sources.
- 2-business-day turnaround, prices and rates are business decisions: see the proposal skeleton.

## Proposal skeleton (fill in your numbers)
| Map type | Fixed price |
|---|---|
| Regional access | $__ |
| Amenity | $__ |
| Aerial parcel | $__ |
| Demographic rings | $__ |

Hourly rate: $__ . Samples: `output/commercial/sample_site__{regional_access,demographic_rings,suitability_retail}.png` (swap in aerial and amenities once imagery and a real list are available).
Acceptance checks automated in `tests/test_commercial.py`: 3300x2550, 300 ppi, PDF and SVG produced. Style/brand match, spelling against the supplied list and parcel fidelity still need a human pass against the client's files.
