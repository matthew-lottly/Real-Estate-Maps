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
        ds = MarketDataset()
        keys = MetricRegistry.keys() if a.metric == "all" else [a.metric]
        for k in keys:
            out = MapComposer(ds, MetricRegistry.get(k), StandardElements.poster(), period_end=a.period).save(paths.output / f"houston_{k}.png", a.dpi)
            C.print(f"[green]✔[/green] {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(CommandLineApp().main())
