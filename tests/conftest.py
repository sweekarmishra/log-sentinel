import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lad.features import build_features  # noqa: E402
from lad.generate import generate_logs  # noqa: E402
from lad.model import Detector  # noqa: E402


@pytest.fixture(scope="session")
def small_world():
    """Small deterministic dataset + a model trained on its training period."""
    events, windows, val_start, test_start = generate_logs(
        seed=3, n_hosts=8, days=2, rate=0.9, n_train_attacks=0, n_val_attacks=0, n_test_attacks=6,
        train_fraction=0.5, val_fraction=0.2)
    feats = build_features(events)
    det = Detector(alert_quantile=0.995, n_estimators=100).fit(feats[feats["minute"] < val_start])
    return {"events": events, "windows": windows, "feats": feats, "det": det, "test_start": test_start}


@pytest.fixture()
def model_path(small_world, tmp_path):
    p = tmp_path / "model.joblib"
    small_world["det"].save(str(p))
    return str(p)
