import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from tests.helpers import build_tiny_bundle, synthetic_clean, synthetic_quikr_raw  # noqa: E402


@pytest.fixture(scope="session")
def clean_df():
    return synthetic_clean(400, 0)


@pytest.fixture(scope="session")
def raw_df():
    return synthetic_quikr_raw(0)


@pytest.fixture(scope="session")
def cfg():
    from carprice.config import load_settings

    return load_settings().cfg


@pytest.fixture(scope="session")
def tiny_bundle_dir(tmp_path_factory):
    return build_tiny_bundle(tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="session")
def predictor(tiny_bundle_dir):
    from carprice.serving.predictor import Predictor

    return Predictor.from_dir(bundle_dir=tiny_bundle_dir)


@pytest.fixture(scope="session")
def real_bundle_dir():
    """The real trained champion, if the pipeline has been run; otherwise tests are skipped."""
    from carprice.models.bundle import BundleError, latest_bundle_dir

    try:
        import os

        return latest_bundle_dir(Path(os.environ.get("CARPRICE_MODELS_DIR", ROOT / "artifacts" / "models")))
    except BundleError:
        pytest.skip("no trained model bundle (run scripts/run_pipeline.py first)")
