"""Regression tests against the REAL trained champion. Skipped automatically when no bundle exists."""
import pytest

from carprice.models import bundle as bnd
from carprice.serving.predictor import Predictor, VehicleInput

pytestmark = pytest.mark.needs_bundle


@pytest.fixture(scope="module")
def real(real_bundle_dir):
    return Predictor.from_dir(bundle_dir=real_bundle_dir)


def V(**kw):
    d = dict(brand="Maruti", model="Swift", year=2015, kms_driven=45000, fuel_type="Petrol", variant="VXi")
    d.update(kw)
    return VehicleInput(**d)


def test_real_bundle_loads_in_strict_mode_and_reproduces_golden(real_bundle_dir):
    b = bnd.load_bundle(real_bundle_dir, strict_env=True, check_golden=True)
    assert b.metadata["test_metrics"]["mae_inr"] > 0


def test_known_examples_are_in_plausible_market_ranges(real):
    swift = real.predict_one(V(), explain=False).predicted_price_inr
    assert 150_000 <= swift <= 700_000  # ~2015 Swift, 45k km: low-single-digit lakh
    audi = real.predict_one(V(brand="Audi", model="Q5", variant="2.0", year=2014, fuel_type="Diesel"), explain=False)
    assert audi.predicted_price_inr > 3 * swift * 0.8


def test_age_and_brand_invariants_hold_on_the_real_model(real):
    prices = [real.predict_one(V(year=y), explain=False).predicted_price_inr for y in (2005, 2012, 2018)]
    assert prices[0] < prices[2]
    assert real.predict_one(V(brand="BMW", model="3 Series", variant="320d", fuel_type="Diesel"), explain=False).predicted_price_inr > \
        real.predict_one(V(brand="Maruti", model="Alto 800", variant="", year=2015), explain=False).predicted_price_inr


def test_intervals_are_wider_than_zero_and_contain_the_point_estimate(real):
    p = real.predict_one(V(), explain=False)
    assert p.lower_inr < p.predicted_price_inr < p.upper_inr
    assert (p.upper_inr - p.lower_inr) / p.predicted_price_inr > 0.1


def test_recorded_interval_coverage_is_reasonable(real):
    iv = real.meta["interval"]
    assert iv["empirical_test_coverage"] >= iv["target_coverage"] - 3 * iv["coverage_se"] - 0.02
