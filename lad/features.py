"""Turn raw log events into per-host, per-minute feature rows."""

from __future__ import annotations

import pandas as pd

FEATURES = [
    "n_events",
    "n_auth",
    "n_auth_fail",
    "auth_fail_ratio",
    "n_unique_users",
    "n_unique_src_ips",
    "n_net",
    "n_unique_dst_ips",
    "n_unique_dst_ports",
    "bytes_out_sum",
    "bytes_out_max",
]

REQUIRED_COLUMNS = ["ts", "host", "event_type"]


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate events to one row per (host, minute).

    Missing optional columns are treated as empty, so the API can accept partial payloads.
    Only minutes in which a host logged something appear in the output.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = events.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="raise")
    for col, default in [("user", ""), ("src_ip", ""), ("dst_ip", ""),
                         ("dst_port", 0), ("bytes_out", 0), ("success", 1)]:
        if col not in df.columns:
            df[col] = default
    df["minute"] = df["ts"].dt.floor("min")

    is_auth = df["event_type"] == "auth"
    is_net = df["event_type"] == "net"
    df["_auth_fail"] = (is_auth & (df["success"] == 0)).astype(int)
    df["_auth"] = is_auth.astype(int)
    df["_net"] = is_net.astype(int)
    df["_user"] = df["user"].where(is_auth)
    df["_src"] = df["src_ip"].where(is_auth)
    df["_dst"] = df["dst_ip"].where(is_net)
    df["_port"] = df["dst_port"].where(is_net)
    df["_bytes"] = df["bytes_out"].where(is_net, 0)

    g = df.groupby(["host", "minute"], sort=True)
    out = pd.DataFrame({
        "n_events": g.size(),
        "n_auth": g["_auth"].sum(),
        "n_auth_fail": g["_auth_fail"].sum(),
        "n_unique_users": g["_user"].nunique(),
        "n_unique_src_ips": g["_src"].nunique(),
        "n_net": g["_net"].sum(),
        "n_unique_dst_ips": g["_dst"].nunique(),
        "n_unique_dst_ports": g["_port"].nunique(),
        "bytes_out_sum": g["_bytes"].sum(),
        "bytes_out_max": g["_bytes"].max(),
    })
    out["auth_fail_ratio"] = (out["n_auth_fail"] / out["n_auth"].where(out["n_auth"] > 0)).fillna(0.0)
    out = out[FEATURES].astype(float).reset_index()
    return out
