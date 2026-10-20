import pytest
from PIL import Image

from real_estate_maps.commercial.context import ContextBuilder
from real_estate_maps.commercial.inputs import SampleSite
from real_estate_maps.config import Paths

pytestmark = pytest.mark.skipif(not (ContextBuilder(Paths()).out / "roads.geojson").exists(), reason="context data missing")


@pytest.fixture(scope="module")
def ctx():
    from real_estate_maps.commercial.base import MapContext
    from real_estate_maps.commercial.brand import Brand
    return MapContext(Brand(), SampleSite.make())


def test_sample_parcel_geometry():
    s = SampleSite.make()
    assert s.acres == pytest.approx(330 * 470 / 43560, rel=0.01)


@pytest.mark.parametrize("name", ["RegionalAccessMap", "DemographicRingsMap", "AerialParcelMap", "SiteLocatorMap", "MarketTrendsMap"])
def test_export_is_print_spec(ctx, tmp_path, name):
    import importlib
    mod = {"RegionalAccessMap": "regional", "DemographicRingsMap": "rings", "AerialParcelMap": "parcel", "SiteLocatorMap": "locator", "MarketTrendsMap": "trends"}[name]
    out = getattr(importlib.import_module(f"real_estate_maps.commercial.{mod}"), name)(ctx).export(tmp_path)
    im = Image.open(out["png"])
    assert im.size == (3300, 2550)
    assert round(im.info["dpi"][0]) == 300
    assert out["pdf"].read_bytes()[:4] == b"%PDF"
    assert 'id="header' in out["svg"].read_text() or "header" in out["svg"].read_text()
