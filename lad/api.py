"""Flask app: web dashboard (/) plus REST API (/detect, /api/*)."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from .baseline import rule_flags
from .evaluate import evaluate_incidents
from .features import FEATURES, REQUIRED_COLUMNS, build_features
from .generate import generate_logs
from .model import Detector, group_incidents

MAX_EVENTS = 200_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC_DIR = Path(__file__).parent / "static"


def load_or_train(path: str) -> Detector:
    """Load the saved model; if it is missing (fresh deploy), train one from synthetic logs."""
    if os.path.exists(path):
        try:
            return Detector.load(path)
        except Exception:  # file saved by a different scikit-learn/pandas version: retrain instead
            pass
    events, _, val_start, _ = generate_logs(seed=7)
    feats = build_features(events)
    det = Detector(alert_quantile=0.999).fit(feats[feats["minute"] < val_start])
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        det.save(path)
    except OSError:  # read-only filesystem: keep the in-memory model
        pass
    return det


def _alerts_json(incidents: pd.DataFrame) -> list[dict]:
    return [
        {
            "host": r.host,
            "start": r.start.isoformat(),
            "end": r.end.isoformat(),
            "minutes_flagged": int(r.n_minutes),
            "score": round(float(r.max_score), 4),
            "top_feature": r.top_feature,
        }
        for r in incidents.itertuples()
    ]


def _heatmap(scored: pd.DataFrame, t0: pd.Timestamp, n_minutes: int) -> dict:
    """Sparse per-host score series for the dashboard heatmap."""
    hosts = sorted(scored["host"].unique())
    series = {}
    for h in hosts:
        g = scored[scored["host"] == h]
        idx = ((g["minute"] - t0) / pd.Timedelta(minutes=1)).astype(int).clip(0, n_minutes - 1)
        series[h] = [[int(i), round(float(s), 3), int(f)] for i, s, f in zip(idx, g["score"], g["flagged"])]
    return {"hosts": hosts, "t0": t0.isoformat(), "n_minutes": int(n_minutes), "series": series}


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    path = model_path or os.environ.get("LAD_MODEL", "artifacts/model.joblib")
    detector = load_or_train(path)

    # ---------- UI ----------
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    # ---------- REST API ----------
    @app.get("/health")
    def health():
        return jsonify(status="ok", features=FEATURES, alert_threshold=-detector.threshold_)

    def _score_events(events: pd.DataFrame):
        feats = build_features(events)
        incidents = detector.detect(feats)
        return feats, incidents

    @app.post("/detect")
    def detect():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("events"), list) or not body["events"]:
            return jsonify(error="body must be JSON: {\"events\": [ ... non-empty list ... ]}"), 400
        if len(body["events"]) > MAX_EVENTS:
            return jsonify(error=f"too many events (max {MAX_EVENTS})"), 413
        try:
            events = pd.DataFrame(body["events"])
            missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
            if missing:
                return jsonify(error=f"events missing required fields: {missing}"), 400
            feats, incidents = _score_events(events)
        except (ValueError, TypeError) as exc:
            return jsonify(error=f"could not parse events: {exc}"), 400
        alerts = _alerts_json(incidents)
        return jsonify(n_events=len(events), n_host_minutes=len(feats), n_alerts=len(alerts), alerts=alerts)

    @app.get("/api/demo")
    def demo():
        """Generate fresh labelled synthetic logs and score the held-out test period."""
        try:
            seed = int(request.args.get("seed", 7))
        except ValueError:
            return jsonify(error="seed must be an integer"), 400
        events, windows, _, test_start = generate_logs(seed=seed)
        feats = build_features(events)
        test = feats[feats["minute"] >= test_start].reset_index(drop=True)
        w_test = windows[windows["split"] == "test"].reset_index(drop=True)

        test["score"] = detector.score(test)
        test["flagged"] = test["score"] > -detector.threshold_
        flagged = test[test["flagged"]].copy()
        if len(flagged):
            flagged["top_feature"] = detector.top_feature(flagged)
            model_inc = group_incidents(flagged)
        else:
            model_inc = pd.DataFrame(columns=["host", "start", "end", "n_minutes", "max_score", "top_feature"])
        m_model = evaluate_incidents(model_inc, w_test)

        rf = rule_flags(test)
        rf = rf[rf["flagged"]].copy()
        rf["score"] = 1.0
        rule_inc = group_incidents(rf) if len(rf) else pd.DataFrame(columns=["host", "start", "end"])
        m_rules = evaluate_incidents(rule_inc, w_test)

        # mark each alert true/false positive using the labels
        alerts = _alerts_json(model_inc)
        for a, r in zip(alerts, model_inc.itertuples()):
            hit = w_test[(w_test["host"] == r.host) & (w_test["start"] < r.end) & (w_test["end"] > r.start)]
            a["true_positive"] = bool(len(hit))
            a["attack_kind"] = hit.iloc[0]["kind"] if len(hit) else None

        n_min = int((events["ts"].max().floor("min") - test_start) / pd.Timedelta(minutes=1)) + 1
        t0 = pd.Timestamp(test_start)
        def pick(m):
            return {k: (None if (isinstance(m[k], float) and np.isnan(m[k])) else m[k])
                    for k in ("attack_windows", "windows_detected", "recall", "incidents", "true_incidents", "precision")}
        return jsonify(
            seed=seed,
            n_events=int(len(events)),
            n_test_host_minutes=int(len(test)),
            model=pick(m_model),
            rules=pick(m_rules),
            alerts=alerts,
            attacks=[{"kind": w.kind, "host": w.host, "start": w.start.isoformat(), "end": w.end.isoformat(),
                      "stealthy": bool(w.stealthy), "detected": bool(d["detected"])}
                     for w, d in zip(w_test.itertuples(), m_model["per_window"])],
            heatmap=_heatmap(test, t0, n_min),
        )

    @app.post("/api/upload")
    def upload():
        """Score a CSV of logs (columns: ts, host, event_type, plus optional user/src_ip/dst_ip/dst_port/bytes_out/success)."""
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="attach a CSV file in the 'file' field"), 400
        try:
            events = pd.read_csv(io.BytesIO(f.read()))
            missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
            if missing:
                return jsonify(error=f"CSV is missing required columns: {missing}"), 400
            if len(events) > MAX_EVENTS:
                return jsonify(error=f"too many rows (max {MAX_EVENTS})"), 413
            events = events.fillna({"user": "", "src_ip": "", "dst_ip": ""}) if "user" in events else events
            feats = build_features(events)
        except Exception as exc:  # malformed CSV / bad timestamps
            return jsonify(error=f"could not read CSV: {exc}"), 400
        feats["score"] = detector.score(feats)
        feats["flagged"] = feats["score"] > -detector.threshold_
        flagged = feats[feats["flagged"]].copy()
        if len(flagged):
            flagged["top_feature"] = detector.top_feature(flagged)
            incidents = group_incidents(flagged)
        else:
            incidents = pd.DataFrame(columns=["host", "start", "end", "n_minutes", "max_score", "top_feature"])
        t0 = feats["minute"].min()
        n_min = int((feats["minute"].max() - t0) / pd.Timedelta(minutes=1)) + 1
        return jsonify(n_events=int(len(events)), n_host_minutes=int(len(feats)),
                       alerts=_alerts_json(incidents), heatmap=_heatmap(feats, t0, n_min))

    @app.get("/api/sample.csv")
    def sample_csv():
        """A small CSV containing a brute-force burst, for trying the upload button."""
        events, windows, _, _ = generate_logs(seed=21, n_hosts=6, days=1, n_train_attacks=0,
                                              n_val_attacks=0, n_test_attacks=3)
        buf = io.StringIO()
        events.to_csv(buf, index=False)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=sample_logs.csv"})

    return app


app = None  # created lazily by wsgi.py


if __name__ == "__main__":  # pragma: no cover
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
