"""End-to-end run: generate logs -> features -> fit on past -> pick threshold on validation ->
score the held-out test period once.

Selection protocol (no peeking at test):
  * IsolationForest is fit on the train period only (unlabelled).
  * The alert quantile is chosen from a fixed grid on the validation period (best incident-level F1).
  * The test period is evaluated once with that choice.
The rule baseline has fixed, analyst-style thresholds (lad/baseline.py) and is never tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lad.baseline import rule_flags
from lad.evaluate import evaluate_incidents
from lad.features import build_features
from lad.generate import generate_logs
from lad.model import Detector, group_incidents

QUANTILE_GRID = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def f1(m: dict) -> float:
    p, r = m["precision"], m["recall"]
    if pd.isna(p) or p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rule_incidents(feats: pd.DataFrame) -> pd.DataFrame:
    flagged = rule_flags(feats)
    flagged = flagged[flagged["flagged"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["host", "start", "end"])
    flagged["score"] = 1.0
    return group_incidents(flagged)


def strip(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "per_window"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    events, windows, val_start, test_start = generate_logs(seed=args.seed)
    print(f"events: {len(events):,}   attack windows: {len(windows)}")

    feats = build_features(events)
    train = feats[feats["minute"] < val_start].reset_index(drop=True)
    val = feats[(feats["minute"] >= val_start) & (feats["minute"] < test_start)].reset_index(drop=True)
    test = feats[feats["minute"] >= test_start].reset_index(drop=True)
    w_val = windows[windows["split"] == "val"].reset_index(drop=True)
    w_test = windows[windows["split"] == "test"].reset_index(drop=True)
    print(f"host-minutes train/val/test: {len(train):,}/{len(val):,}/{len(test):,}   "
          f"attack windows val/test: {len(w_val)}/{len(w_test)}")

    # choose alert quantile on validation
    sweep = []
    for q in QUANTILE_GRID:
        det_q = Detector(alert_quantile=q).fit(train)
        m = evaluate_incidents(det_q.detect(val), w_val)
        sweep.append({"alert_quantile": q, "val_recall": m["recall"], "val_precision": m["precision"],
                      "val_incidents": m["incidents"], "val_f1": f1(m)})
    sweep_df = pd.DataFrame(sweep)
    best_q = float(sweep_df.sort_values(["val_f1", "alert_quantile"], ascending=[False, False]).iloc[0]["alert_quantile"])
    print("\nvalidation sweep:\n" + sweep_df.round(3).to_string(index=False))
    print(f"chosen alert_quantile = {best_q}")

    det = Detector(alert_quantile=best_q).fit(train)
    model_inc = det.detect(test)
    m_model = evaluate_incidents(model_inc, w_test)
    rules_inc = rule_incidents(test)
    m_rules = evaluate_incidents(rules_inc, w_test)
    m_rules_val = evaluate_incidents(rule_incidents(val), w_val)

    def by_kind(m: dict) -> dict:
        pw = pd.DataFrame(m["per_window"]).merge(w_test[["attack_id", "stealthy"]], on="attack_id")
        g = pw.groupby("kind")["detected"].agg(["sum", "count"])
        s = pw.groupby("stealthy")["detected"].agg(["sum", "count"])
        return {
            "by_kind": {k: f"{int(r['sum'])}/{int(r['count'])}" for k, r in g.iterrows()},
            "by_intensity": {("stealthy" if k else "obvious"): f"{int(r['sum'])}/{int(r['count'])}"
                             for k, r in s.iterrows()},
        }

    summary = {
        "n_events": int(len(events)),
        "chosen_alert_quantile": best_q,
        "validation_sweep": sweep,
        "test": {
            "isolation_forest": {**strip(m_model), **by_kind(m_model)},
            "rule_baseline": {**strip(m_rules), **by_kind(m_rules)},
        },
        "validation_rule_baseline": strip(m_rules_val),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    det.save(str(out / "model.joblib"))
    events.head(300).to_csv(out / "sample_events.csv", index=False)
    windows.to_csv(out / "attack_windows.csv", index=False)
    model_inc.to_csv(out / "model_incidents.csv", index=False)

    for name in ("isolation_forest", "rule_baseline"):
        print(f"\n== TEST: {name} ==")
        for k, v in summary["test"][name].items():
            print(f"  {k}: {v}")
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
