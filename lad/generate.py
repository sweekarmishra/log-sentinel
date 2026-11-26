"""Synthetic network + authentication log generator with labelled attack windows.

Event schema (one row per log event):
    ts          datetime64  event time
    host        str         machine that produced the log (auth target / network source)
    event_type  str         "auth" or "net"
    user        str         account name (auth events only)
    src_ip      str         remote address (auth events only)
    dst_ip      str         destination address (net events only)
    dst_port    int         destination port (net events only)
    bytes_out   int         bytes sent (net events only)
    success     int         1/0 login result (auth events only)

Realism notes
-------------
* Benign traffic has confounders that look like attacks to a naive threshold: nightly backup uploads,
  a vulnerability scanner sweeping ports, and bursts of mistyped passwords.
* Half of the attacks are "stealthy" (lower intensity), so fixed thresholds can miss them.
Attack labels are returned separately and never reach the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "ts", "host", "event_type", "user", "src_ip", "dst_ip", "dst_port", "bytes_out", "success",
]

COMMON_PORTS = np.array([53, 80, 123, 443, 443, 443, 8080, 22, 3306])
INTERNAL_SERVERS = [f"10.0.1.{i}" for i in range(1, 9)]
EXTERNAL_SITES = [f"34.120.{i}.{j}" for i in range(1, 6) for j in range(1, 6)]
USERS = [f"user{i:02d}" for i in range(1, 31)]
START = pd.Timestamp("2026-01-05 00:00:00")


def _diurnal(minute_of_day: np.ndarray) -> np.ndarray:
    """Activity multiplier: quiet at night, busy mid-day."""
    return 0.35 + 0.65 * np.clip(np.sin((minute_of_day - 6 * 60) / (24 * 60) * 2 * np.pi) * 0.5 + 0.5, 0, 1)


def _empty(ts, host, event_type) -> pd.DataFrame:
    n = len(ts)
    return pd.DataFrame({
        "ts": ts, "host": host, "event_type": event_type,
        "user": "", "src_ip": "", "dst_ip": "", "dst_port": 0, "bytes_out": 0, "success": 0,
    }, index=range(n))


def _benign(rng: np.random.Generator, n_hosts: int, n_minutes: int, rate: float) -> pd.DataFrame:
    hosts = [f"host{h:02d}" for h in range(n_hosts)]
    host_scale = rng.lognormal(mean=0.0, sigma=0.5, size=n_hosts)
    host_dsts = {h: rng.choice(INTERNAL_SERVERS + EXTERNAL_SITES, size=6, replace=False) for h in hosts}
    minutes = np.arange(n_minutes)
    mult = _diurnal(minutes % (24 * 60))

    rows = []
    for hi, host in enumerate(hosts):
        counts = rng.poisson(rate * host_scale[hi] * mult)
        total = int(counts.sum())
        if total == 0:
            continue
        minute_idx = np.repeat(minutes, counts)
        ts = START + pd.to_timedelta(minute_idx * 60 + rng.uniform(0, 60, size=total), unit="s")
        is_auth = rng.random(total) < 0.18
        n_auth = int(is_auth.sum())
        n_net = total - n_auth

        df = _empty(ts, host, np.where(is_auth, "auth", "net"))
        df.loc[is_auth, "user"] = rng.choice(USERS[:10], size=n_auth)
        df.loc[is_auth, "src_ip"] = rng.choice(INTERNAL_SERVERS, size=n_auth)
        df.loc[is_auth, "success"] = (rng.random(n_auth) > 0.04).astype(int)
        df.loc[~is_auth, "dst_ip"] = rng.choice(host_dsts[host], size=n_net)
        df.loc[~is_auth, "dst_port"] = rng.choice(COMMON_PORTS, size=n_net)
        df.loc[~is_auth, "bytes_out"] = np.minimum(
            rng.lognormal(mean=8.0, sigma=1.6, size=n_net), 5e7).astype(np.int64)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _benign_confounders(rng: np.random.Generator, n_hosts: int, days: int) -> pd.DataFrame:
    """Legitimate activity that resembles attacks."""
    parts = []
    # 1. nightly backup upload from two hosts: large transfers to an internal backup server
    for host in ("host03", "host11"):
        if int(host[-2:]) >= n_hosts:
            continue
        for d in range(days):
            for m in range(d * 1440 + 120, d * 1440 + 135):  # 02:00-02:15
                n = int(rng.integers(3, 8))
                df = _empty(START + pd.Timedelta(minutes=m) + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"),
                            host, "net")
                df["dst_ip"] = "10.0.1.8"
                df["dst_port"] = 445
                df["bytes_out"] = rng.integers(2_000_000, 6_000_000, size=n)
                parts.append(df)
    # 2. vulnerability scanner sweeping ports for a few minutes each night
    if n_hosts > 7:
        for d in range(days):
            start = d * 1440 + 180
            for m in range(start, start + int(rng.integers(2, 5))):
                n = int(rng.integers(60, 160))
                df = _empty(START + pd.Timedelta(minutes=m) + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"),
                            "host07", "net")
                df["dst_ip"] = f"10.0.1.{int(rng.integers(1, 9))}"
                df["dst_port"] = rng.integers(1, 4000, size=n)
                df["bytes_out"] = rng.integers(40, 120, size=n)
                parts.append(df)
    # 3. users mistyping passwords: short bursts of failed logins on random hosts
    n_bursts = days * 6
    for _ in range(n_bursts):
        host = f"host{int(rng.integers(0, n_hosts)):02d}"
        m = int(rng.integers(0, days * 1440 - 3))
        n = int(rng.integers(6, 16))
        df = _empty(START + pd.Timedelta(minutes=m) + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"),
                    host, "auth")
        df["user"] = USERS[int(rng.integers(0, 10))]
        df["src_ip"] = INTERNAL_SERVERS[int(rng.integers(0, 8))]
        df["success"] = 0
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _attack_events(kind: str, host: str, start_min: int, dur: int, stealthy: bool,
                   rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    src_attacker = f"203.0.113.{int(rng.integers(2, 250))}"
    target = f"10.0.2.{int(rng.integers(2, 250))}"
    for m in range(start_min, start_min + dur):
        base = START + pd.Timedelta(minutes=m)
        if kind == "brute_force":
            n = int(rng.integers(12, 30)) if stealthy else int(rng.integers(40, 120))
            d = _empty(base + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"), host, "auth")
            d["user"] = rng.choice(USERS, size=n)
            d["src_ip"] = src_attacker
            d["success"] = (rng.random(n) < 0.01).astype(int)
        elif kind == "port_scan":
            n = int(rng.integers(30, 90)) if stealthy else int(rng.integers(120, 400))
            d = _empty(base + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"), host, "net")
            d["dst_ip"] = target
            d["dst_port"] = rng.integers(1, 10000, size=n)
            d["bytes_out"] = rng.integers(40, 120, size=n)
        elif kind == "exfiltration":
            if stealthy:
                n = int(rng.integers(4, 10))
                size = rng.integers(800_000, 2_000_000, size=n)
            else:
                n = int(rng.integers(8, 30))
                size = rng.integers(3_000_000, 15_000_000, size=n)
            d = _empty(base + pd.to_timedelta(rng.uniform(0, 60, n), unit="s"), host, "net")
            d["dst_ip"] = "198.51.100.77"
            d["dst_port"] = 443
            d["bytes_out"] = size
        else:  # pragma: no cover
            raise ValueError(kind)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def generate_logs(
    seed: int = 7,
    n_hosts: int = 20,
    days: int = 3,
    rate: float = 0.95,
    n_train_attacks: int = 3,
    n_val_attacks: int = 6,
    n_test_attacks: int = 13,
    train_fraction: float = 0.5,
    val_fraction: float = 0.2,
    confounders: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Return (events, attack_windows, val_start, test_start).

    Time is cut into train | validation | test. Hyper-parameters are chosen on validation only;
    the test period is scored once. The model never sees labels.
    """
    rng = np.random.default_rng(seed)
    n_minutes = days * 24 * 60
    val_min = int(n_minutes * train_fraction)
    test_min = int(n_minutes * (train_fraction + val_fraction))
    val_start = START + pd.Timedelta(minutes=val_min)
    test_start = START + pd.Timedelta(minutes=test_min)

    events = [_benign(rng, n_hosts, n_minutes, rate)]
    if confounders:
        events.append(_benign_confounders(rng, n_hosts, days))

    hosts = [f"host{h:02d}" for h in range(n_hosts)]
    kinds = ["brute_force", "port_scan", "exfiltration"]
    windows: list[dict] = []

    def place(n: int, lo: int, hi: int, split: str) -> None:
        slots = np.linspace(lo + 20, hi - 40, n + 1).astype(int)
        for i in range(n):
            kind = kinds[len(windows) % 3]
            host = hosts[int(rng.integers(0, n_hosts))]
            dur = {"brute_force": int(rng.integers(3, 8)),
                   "port_scan": int(rng.integers(2, 6)),
                   "exfiltration": int(rng.integers(5, 14))}[kind]
            stealthy = bool(rng.random() < 0.5)
            start_min = int(rng.integers(slots[i], max(slots[i] + 1, slots[i + 1] - dur - 5)))
            events.append(_attack_events(kind, host, start_min, dur, stealthy, rng))
            windows.append({
                "attack_id": len(windows), "kind": kind, "host": host, "stealthy": stealthy,
                "start": START + pd.Timedelta(minutes=start_min),
                "end": START + pd.Timedelta(minutes=start_min + dur),
                "split": split,
            })

    place(n_train_attacks, 0, val_min, "train")
    place(n_val_attacks, val_min, test_min, "val")
    place(n_test_attacks, test_min, n_minutes, "test")

    df = pd.concat(events, ignore_index=True)
    df = df[EVENT_COLUMNS].sort_values("ts").reset_index(drop=True)
    return df, pd.DataFrame(windows), val_start, test_start
