"""Static-threshold rule baseline, the way an analyst would write it without a model."""

from __future__ import annotations

import pandas as pd

RULES = {
    "brute_force": ("n_auth_fail", 20),          # failed logins in one minute
    "port_scan": ("n_unique_dst_ports", 50),     # distinct destination ports in one minute
    "exfiltration": ("bytes_out_sum", 20_000_000),  # bytes sent in one minute (~20 MB)
}


def rule_flags(feats: pd.DataFrame) -> pd.DataFrame:
    """Return feats with one boolean column per rule plus ``flagged`` (any rule) and ``rule``."""
    out = feats.copy()
    for name, (col, thr) in RULES.items():
        out[f"rule_{name}"] = out[col] >= thr
    rule_cols = [f"rule_{n}" for n in RULES]
    out["flagged"] = out[rule_cols].any(axis=1)
    out["rule"] = out[rule_cols].idxmax(axis=1).str.replace("rule_", "", regex=False)
    out.loc[~out["flagged"], "rule"] = ""
    return out
