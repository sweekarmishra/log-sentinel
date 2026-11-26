"""Unsupervised IsolationForest detector with per-alert feature attribution."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, build_features


def _transform(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """log1p compresses the heavy tails of counts and byte volumes."""
    return np.log1p(np.asarray(X, dtype=float))


@dataclass
class Detector:
    n_estimators: int = 300
    alert_quantile: float = 0.995   # fraction of training rows treated as normal
    random_state: int = 0
    features: list = field(default_factory=lambda: list(FEATURES))
    scaler: StandardScaler | None = None
    forest: IsolationForest | None = None
    threshold_: float | None = None
    medians_: np.ndarray | None = None  # per-feature "typical" value (scaled space)

    # ---- training ----
    def fit(self, feats: pd.DataFrame) -> "Detector":
        Z = _transform(feats[self.features])
        self.scaler = StandardScaler().fit(Z)
        Zs = self.scaler.transform(Z)
        self.forest = IsolationForest(
            n_estimators=self.n_estimators, contamination="auto", random_state=self.random_state, n_jobs=1
        ).fit(Zs)
        scores = self.forest.score_samples(Zs)           # lower = more anomalous
        # Flag anything more anomalous than all but the lowest (1 - alert_quantile) of training rows.
        self.threshold_ = float(np.quantile(scores, 1 - self.alert_quantile))
        self.medians_ = np.median(Zs, axis=0)
        return self

    # ---- inference ----
    def _scaled(self, feats: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(_transform(feats[self.features]))

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        """Anomaly score; higher = more anomalous (negated sklearn score)."""
        return -self.forest.score_samples(self._scaled(feats))

    def is_anomaly(self, feats: pd.DataFrame) -> np.ndarray:
        return self.score(feats) > -self.threshold_

    def top_feature(self, feats: pd.DataFrame) -> list[str]:
        """Feature whose reset to the training median lowers the anomaly score the most.

        For each row, replace one feature at a time with its typical value and re-score;
        the feature giving the largest drop is the top contributor. Cost: n_features extra
        scoring passes, only run on rows that were flagged.
        """
        Zs = self._scaled(feats)
        base = -self.forest.score_samples(Zs)
        drops = np.zeros((Zs.shape[0], len(self.features)))
        for j in range(len(self.features)):
            Zj = Zs.copy()
            Zj[:, j] = self.medians_[j]
            drops[:, j] = base + self.forest.score_samples(Zj)  # base - new
        return [self.features[i] for i in drops.argmax(axis=1)]

    # ---- alerts ----
    def detect(self, feats: pd.DataFrame, gap_minutes: int = 2) -> pd.DataFrame:
        """Flag host-minutes and merge nearby flags on the same host into incidents."""
        feats = feats.reset_index(drop=True)
        feats["score"] = self.score(feats)
        feats["flagged"] = feats["score"] > -self.threshold_
        flagged = feats[feats["flagged"]].copy()
        if flagged.empty:
            return pd.DataFrame(columns=["host", "start", "end", "n_minutes", "max_score", "top_feature"])
        flagged["top_feature"] = self.top_feature(flagged)
        return group_incidents(flagged, gap_minutes)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Detector":
        return joblib.load(path)


def group_incidents(flagged: pd.DataFrame, gap_minutes: int = 2) -> pd.DataFrame:
    """Merge flagged host-minutes into incidents (same host, gaps <= gap_minutes)."""
    out = []
    for host, g in flagged.sort_values(["host", "minute"]).groupby("host"):
        g = g.reset_index(drop=True)
        gaps = g["minute"].diff() > pd.Timedelta(minutes=gap_minutes)
        gid = gaps.cumsum()
        for _, inc in g.groupby(gid):
            worst = inc.loc[inc["score"].idxmax()]
            out.append({
                "host": host,
                "start": inc["minute"].min(),
                "end": inc["minute"].max() + pd.Timedelta(minutes=1),
                "n_minutes": len(inc),
                "max_score": float(inc["score"].max()),
                "top_feature": worst.get("top_feature", ""),
            })
    return pd.DataFrame(out).sort_values("start").reset_index(drop=True)


def detect_from_events(detector: Detector, events: pd.DataFrame) -> pd.DataFrame:
    return detector.detect(build_features(events))
