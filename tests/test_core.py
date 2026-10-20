import numpy as np
import pytest

from real_estate_maps.classification import ClassifierFactory
from real_estate_maps.metrics import Fmt, MetricRegistry


def test_formats():
    assert Fmt.pct_signed(0.001) == "0%"
    assert Fmt.pct_signed(-0.404) == "-40%"
    assert Fmt.pct_signed(0.98) == "+98%"


def test_registry_has_metrics():
    assert {"price", "ppsf", "yoy", "dom"} <= set(MetricRegistry.keys())


@pytest.mark.parametrize("key", ["price", "yoy"])
def test_scheme_assigns_every_value(key):
    m = MetricRegistry.get(key)
    vals = np.random.default_rng(1).lognormal(12, 0.5, 200) if key == "price" else np.random.default_rng(1).normal(0, 0.1, 200)
    s = ClassifierFactory.for_metric(m).scheme(vals)
    cls = s.assign(vals)
    assert cls.min() >= 0 and cls.max() < s.n
    assert sum(s.counts(vals)) == len(vals)
