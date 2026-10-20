import pytest

from real_estate_maps.config import Paths

pytestmark = pytest.mark.skipif(not Paths().market_csv.exists(), reason="processed data missing")


def test_snapshot_and_render(tmp_path):
    from real_estate_maps.composer import MapComposer
    from real_estate_maps.data import MarketDataset
    from real_estate_maps.elements import StandardElements
    from real_estate_maps.metrics import MetricRegistry

    ds = MarketDataset()
    assert ds.snapshot().geometry.notna().all()
    out = MapComposer(ds, MetricRegistry.get("price"), StandardElements.poster()).save(tmp_path / "m.png", dpi=40)
    assert out.stat().st_size > 10_000
