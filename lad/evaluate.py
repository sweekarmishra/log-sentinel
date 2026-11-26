"""Window-level recall and incident-level precision against labelled attack windows."""

from __future__ import annotations

import pandas as pd


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def evaluate_incidents(incidents: pd.DataFrame, windows: pd.DataFrame) -> dict:
    """Score a set of incidents (host, start, end) against attack windows.

    recall    = attack windows overlapped by >= 1 incident on the same host / all windows
    precision = incidents overlapping >= 1 attack window on the same host / all incidents
    """
    detected = 0
    per_window = []
    for w in windows.itertuples():
        hit = any(
            i.host == w.host and _overlaps(i.start, i.end, w.start, w.end)
            for i in incidents.itertuples()
        )
        detected += hit
        per_window.append({"attack_id": w.attack_id, "kind": w.kind, "detected": bool(hit)})

    true_incidents = 0
    for i in incidents.itertuples():
        true_incidents += any(
            i.host == w.host and _overlaps(i.start, i.end, w.start, w.end)
            for w in windows.itertuples()
        )
    n_inc = len(incidents)
    return {
        "attack_windows": int(len(windows)),
        "windows_detected": int(detected),
        "recall": detected / len(windows) if len(windows) else float("nan"),
        "incidents": int(n_inc),
        "true_incidents": int(true_incidents),
        "false_incidents": int(n_inc - true_incidents),
        "precision": true_incidents / n_inc if n_inc else float("nan"),
        "per_window": per_window,
    }
