"""Command line: fetch | build-data | render"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .composer import MapComposer
from .config import Paths
from .data import BoundarySource, DatasetBuilder, MarketDataset, RedfinTracker
from .elements import StandardElements
from .metrics import MetricRegistry

C = Console()


class CommandLineApp:
    def parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="real_estate_maps")
        sub = p.add_subparsers(dest="cmd", required=True)
        sub.add_parser("fetch", help="download the Houston rows of Redfin's ZIP tracker (streams ~1.5 GB) and the boundary files into data/raw/")
        sub.add_parser("build-data", help="turn data/raw/ into the small committed files in data/processed/")
        sub.add_parser("build-context", help="download and clip highways, cities, airports and water (Natural Earth) for the commercial maps")
        sub.add_parser("fetch-extra", help="fetch Realtor.com, Overture Maps and USGS elevation data (public, no key)")
        sub.add_parser("build-static", help="ZIP-level POI, road and elevation features from the extra data")
        sub.add_parser("forecast", help="build features, run the walk-forward backtest and write the ZIP forecast files")
        k = sub.add_parser("commercial", help="render the commercial real estate map set (PNG 3300x2550 @300ppi, PDF, layered SVG)")
        k.add_argument("--map", default="all", help="regional, amenities, parcel, rings, neighborhood, locator, trends, suitability-retail, suitability-industrial or all")
        k.add_argument("--inputs", default="inputs/sample", help="folder with brand.json, amenities.csv, optional demographics.csv / aerial.png+json / parcel.geojson")
        r = sub.add_parser("render", help="render poster maps to output/")
        r.add_argument("--metric", default="price", help=f"one of {', '.join(MetricRegistry.keys())} or 'all'")
        r.add_argument("--dpi", type=int, default=160)
        r.add_argument("--period", help="period end YYYY-MM-DD (default: latest)")
        return p

    def main(self, argv=None) -> int:
        a = self.parser().parse_args(argv)
        paths = Paths()
        if a.cmd == "fetch":
            n = RedfinTracker().download(paths.raw / "redfin_houston.tsv")
            BoundarySource().download_all(paths.raw)
            C.print(f"[green]✔[/green] kept {n:,} Houston rows in {paths.raw}")
            return 0
        if a.cmd == "build-data":
            rep = DatasetBuilder().build(paths.raw / "redfin_houston.tsv", paths.raw / "tx_zcta.json", paths.raw / "us_counties.json")
            C.print(f"[green]✔[/green] {rep.rows:,} rows, {rep.zips} ZIPs ({rep.zips_with_polygon} with a 2010 polygon), {rep.periods} periods, latest {rep.latest_end}")
            if rep.missing_polygons:
                C.print(f"[yellow]no 2010 polygon for {len(rep.missing_polygons)} ZIPs:[/yellow] {', '.join(rep.missing_polygons)}")
            return 0
        if a.cmd == "fetch-extra":
            from .forecast.sources import ExtraSources
            e = ExtraSources(paths)
            e.fetch_rdc(); e.fetch_overture(); e.fetch_dem()
            C.print("[green]✔[/green] extra sources ready")
            return 0
        if a.cmd == "build-static":
            from .forecast.geo_features import StaticFeatures
            C.print(f"[green]✔[/green] {StaticFeatures(paths).build().shape}")
            return 0
        if a.cmd == "forecast":
            from .forecast.pipeline import ForecastPipeline
            ForecastPipeline(paths).run()
            C.print("[green]✔[/green] forecast written to data/processed/forecast")
            return 0
        if a.cmd == "build-context":
            from .commercial.context import ContextBuilder
            cb = ContextBuilder(paths)
            cb.fetch()
            C.print(f"[green]✔[/green] {cb.build()}")
            return 0
        if a.cmd == "commercial":
            return self.commercial(a, paths)
        ds = MarketDataset()
        keys = MetricRegistry.keys() if a.metric == "all" else [a.metric]
        for k in keys:
            out = MapComposer(ds, MetricRegistry.get(k), StandardElements.poster(), period_end=a.period).save(paths.output / f"houston_{k}.png", a.dpi)
            C.print(f"[green]✔[/green] {out}")
        return 0

    def commercial(self, a, paths) -> int:
        from .commercial.amenity import AmenityMap
        from .commercial.base import MapContext
        from .commercial.brand import Brand
        from .commercial.inputs import AmenityList, DemographicProvider, SampleSite, Site
        from .commercial.neighborhood import NeighborhoodMap
        from .commercial.parcel import AerialParcelMap
        from .commercial.providers import ImageFileImagery
        from .commercial.regional import RegionalAccessMap
        from .commercial.rings import DemographicRingsMap
        from .commercial.investor import EvidencePage, ForecastMap, RiskReturnMap
        from .commercial.locator import SiteLocatorMap
        from .commercial.suitability import SuitabilityMap
        from .commercial.trends import MarketTrendsMap

        folder = Path(a.inputs)
        brand = Brand.load(folder / "brand.json") if (folder / "brand.json").exists() else Brand()
        parcel = folder / "parcel.geojson"
        site = Site.from_geojson("Subject Site", "Address from client", parcel) if parcel.exists() else SampleSite.make()
        from .commercial.inputs import OvertureAmenities
        amen = AmenityList.load(folder / "amenities.csv") if (folder / "amenities.csv").exists() else OvertureAmenities.around(site, paths.raw)
        ctx = MapContext(brand, site, amen, DemographicProvider(folder / "demographics.csv"), ImageFileImagery.find(folder), paths=paths)
        maps = {"regional": RegionalAccessMap, "amenities": AmenityMap, "parcel": AerialParcelMap, "rings": DemographicRingsMap, "neighborhood": NeighborhoodMap, "locator": SiteLocatorMap, "forecast-24": lambda c: ForecastMap(c, 24), "forecast-12": lambda c: ForecastMap(c, 12), "risk-return": RiskReturnMap, "evidence": EvidencePage, "trends": MarketTrendsMap,
                "suitability-retail": lambda c: SuitabilityMap(c, "retail"), "suitability-industrial": lambda c: SuitabilityMap(c, "industrial")}
        for k in (maps if a.map == "all" else [a.map]):
            out = maps[k](ctx).export(paths.output / "commercial")
            C.print(f"[green]✔[/green] {out['png']}")
            for i in out["issues"]:
                C.print(f"   [yellow]layout:[/yellow] {i}", highlight=False)
        return 0


if __name__ == "__main__":
    raise SystemExit(CommandLineApp().main())
