import pandas as pd
import pytest

from lad.api import create_app
from lad.baseline import rule_flags
from lad.features import FEATURES, build_features
from lad.generate import generate_logs


def _events_for(window, events):
    return events[(events["host"] == window.host) & (events["ts"] >= window.start) & (events["ts"] < window.end)]


# 1 ---------------------------------------------------------------------------
def test_features_aggregate_per_host_per_minute():
    ev = pd.DataFrame([
        {"ts": "2026-01-05 10:00:05", "host": "a", "event_type": "auth", "user": "u1", "src_ip": "1.1.1.1", "success": 0},
        {"ts": "2026-01-05 10:00:20", "host": "a", "event_type": "auth", "user": "u2", "src_ip": "1.1.1.1", "success": 1},
        {"ts": "2026-01-05 10:00:40", "host": "a", "event_type": "net", "dst_ip": "9.9.9.9", "dst_port": 443, "bytes_out": 1000},
        {"ts": "2026-01-05 10:00:50", "host": "a", "event_type": "net", "dst_ip": "9.9.9.9", "dst_port": 80, "bytes_out": 3000},
        {"ts": "2026-01-05 10:01:10", "host": "b", "event_type": "net", "dst_ip": "8.8.8.8", "dst_port": 53, "bytes_out": 50},
    ])
    f = build_features(ev)
    assert len(f) == 2 and list(f.columns[2:]) == FEATURES
    a = f[f["host"] == "a"].iloc[0]
    assert a["n_events"] == 4 and a["n_auth"] == 2 and a["n_auth_fail"] == 1
    assert a["auth_fail_ratio"] == 0.5 and a["n_unique_users"] == 2
    assert a["n_unique_dst_ports"] == 2 and a["bytes_out_sum"] == 4000 and a["bytes_out_max"] == 3000


# 2 ---------------------------------------------------------------------------
def test_generator_is_reproducible_and_labels_attacks():
    e1, w1, _, t1 = generate_logs(seed=11, n_hosts=6, days=2)
    e2, w2, _, t2 = generate_logs(seed=11, n_hosts=6, days=2)
    assert e1.equals(e2) and w1.equals(w2)
    assert set(w1["kind"]) == {"brute_force", "port_scan", "exfiltration"}
    assert (w1[w1["split"] == "test"]["start"] >= t1).all()   # held-out windows really are after the split
    assert (w1["end"] > w1["start"]).all()


# 3 ---------------------------------------------------------------------------
def test_rule_baseline_flags_each_obvious_attack_type():
    base = dict.fromkeys(FEATURES, 0.0)
    rows = pd.DataFrame([
        {**base, "host": "h", "minute": pd.Timestamp("2026-01-05"), "n_auth_fail": 50.0},
        {**base, "host": "h", "minute": pd.Timestamp("2026-01-05 00:01"), "n_unique_dst_ports": 300.0},
        {**base, "host": "h", "minute": pd.Timestamp("2026-01-05 00:02"), "bytes_out_sum": 5e7},
        {**base, "host": "h", "minute": pd.Timestamp("2026-01-05 00:03"), "n_events": 3.0},
    ])
    out = rule_flags(rows)
    assert list(out["rule"]) == ["brute_force", "port_scan", "exfiltration", ""]


# 4 ---------------------------------------------------------------------------
def test_model_flags_an_obvious_attack_and_names_a_sensible_feature(small_world):
    det, feats = small_world["det"], small_world["feats"]
    w = small_world["windows"]
    brute = w[(w["kind"] == "brute_force")].iloc[0]
    rows = feats[(feats["host"] == brute.host) & (feats["minute"] >= brute.start) & (feats["minute"] < brute.end)]
    assert len(rows) > 0
    benign_median = float(pd.Series(det.score(feats)).median())
    assert det.score(rows).max() > benign_median          # attack scores above typical host-minute
    inc = det.detect(feats)
    hit = inc[(inc["host"] == brute.host) & (inc["start"] < brute.end) & (inc["end"] > brute.start)]
    assert len(hit) >= 1
    assert hit.iloc[0]["top_feature"] in FEATURES


# 5 ---------------------------------------------------------------------------
def test_api_detect_returns_alert_with_top_feature(small_world, model_path):
    w = small_world["windows"]
    brute = w[w["kind"] == "brute_force"].iloc[0]
    ev = _events_for(brute, small_world["events"]).copy()
    ev["ts"] = ev["ts"].astype(str)
    client = create_app(model_path).test_client()
    r = client.post("/detect", json={"events": ev.to_dict("records")})
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_alerts"] >= 1
    alert = body["alerts"][0]
    assert alert["host"] == brute.host and alert["top_feature"] in FEATURES


# 7 ---------------------------------------------------------------------------
def test_dashboard_serves_and_upload_scores_csv(small_world, model_path):
    import io
    client = create_app(model_path).test_client()
    page = client.get("/")
    assert page.status_code == 200 and b"Log Sentinel" in page.data
    w = small_world["windows"]
    brute = w[w["kind"] == "brute_force"].iloc[0]
    ev = _events_for(brute, small_world["events"])
    buf = io.StringIO()
    ev.to_csv(buf, index=False)
    r = client.post("/api/upload", data={"file": (io.BytesIO(buf.getvalue().encode()), "logs.csv")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200 and len(body["alerts"]) >= 1 and brute.host in body["heatmap"]["hosts"]
    bad = client.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    assert client.post("/api/upload", data={}, content_type="multipart/form-data").status_code == 400


# 8 ---------------------------------------------------------------------------
def test_demo_endpoint_reports_labelled_metrics(model_path):
    client = create_app(model_path).test_client()
    r = client.get("/api/demo?seed=5")
    d = r.get_json()
    assert r.status_code == 200
    assert d["model"]["attack_windows"] == 13 and d["rules"]["attack_windows"] == 13
    assert all("true_positive" in a for a in d["alerts"])
    assert client.get("/api/demo?seed=notanumber").status_code == 400


# 6 ---------------------------------------------------------------------------
def test_api_health_and_input_validation(model_path):
    client = create_app(model_path).test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    assert client.post("/detect", json={}).status_code == 400
    assert client.post("/detect", json={"events": []}).status_code == 400
    assert client.post("/detect", json={"events": [{"host": "a"}]}).status_code == 400
    assert client.post("/detect", data="not json", content_type="text/plain").status_code == 400
