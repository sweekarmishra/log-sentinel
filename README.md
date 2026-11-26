# Log Anomaly and Threat Detection

Converts network and authentication log events into per-host, per-minute features and flags
brute-force, port-scan and data-exfiltration behaviour with an unsupervised IsolationForest.
Detection is exposed through a Flask REST API that returns alerts with the top contributing feature.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py          # (optional) generate logs, train, evaluate, save artifacts/model.joblib
pytest -q                       # 8 tests
python -m lad.api               # web app at http://127.0.0.1:5000
```

The web app (`/`) has a live demo (fresh synthetic logs per seed, model vs rule baseline, host-by-time
score heatmap, alert table marked true/false positive) and a CSV upload for your own logs.
If `artifacts/model.joblib` is missing, the app trains a model on first start.

## Deploy

* **Render (free):** push the folder to GitHub, then New > Blueprint, pick the repo. `render.yaml` is included.
  Free instances sleep when idle, so the first request after a pause takes a while.
* **Hugging Face Spaces:** create a Docker Space and push the folder. The `Dockerfile` listens on port 7860.
* **Anywhere with Docker:** `docker build -t log-sentinel . && docker run -p 7860:7860 log-sentinel`.
* **Railway/Heroku-style:** `Procfile` is included.

The gunicorn start command was not run in the environment this was built in (gunicorn was not installed
there). The same app was run with Flask's server and exercised over real HTTP; run `gunicorn wsgi:app` once locally to confirm.

```bash
curl -s localhost:5000/detect -H 'content-type: application/json' \
  -d '{"events":[{"ts":"2026-01-06 10:00:05","host":"h1","event_type":"auth","user":"root","src_ip":"203.0.113.5","success":0}]}'
```

Response: `{"n_alerts": N, "alerts": [{"host", "start", "end", "minutes_flagged", "score", "top_feature"}]}`.

## How it works

| Stage | File | Notes |
|---|---|---|
| Synthetic logs | `lad/generate.py` | ~59.5k events, 20 hosts, 3 days, 22 labelled attack windows (3 brute-force/port-scan/exfiltration mixes), half "stealthy". Benign confounders: nightly backups, a vuln scanner, mistyped-password bursts. |
| Features | `lad/features.py` | 11 features per (host, minute): event counts, failed logins and ratio, unique users/IPs/ports, bytes out sum/max. |
| Model | `lad/model.py` | log1p + standardise, `IsolationForest`, threshold = quantile of training scores. Top feature = the one whose reset to its training median lowers the anomaly score most. Flagged minutes merge into incidents (same host, gap <= 2 min). |
| Baseline | `lad/baseline.py` | Fixed thresholds: >=20 failed logins/min, >=50 distinct ports/min, >=20 MB out/min. Never tuned. |
| Evaluation | `lad/evaluate.py` | Recall = attack windows overlapped by an incident on the same host. Precision = incidents overlapping a real attack. |

Protocol: time split train 50% / validation 20% / test 30%. The model is fit on train only
(no labels). The alert quantile is picked on validation. The test period (13 attack windows) is scored once.

## Measured results (seed 7, test period only)

| | Windows detected | Recall | Incidents | Precision |
|---|---|---|---|---|
| IsolationForest | 6 / 13 | 0.46 | 16 | 0.38 |
| Rule baseline | 11 / 13 | 0.85 | 13 | 0.92 |

IsolationForest by type: brute-force 5/5, exfiltration 1/4, port-scan 0/4.
Rule baseline by type: brute-force 5/5, exfiltration 2/4, port-scan 4/4.

**On this benchmark the model does not beat the rule baseline.** Reproduce with `python run_pipeline.py`;
exact numbers are in `artifacts/metrics.json`.

## Limitations (read before quoting any number)

* **Synthetic data.** Attacks are injected by the same code that defines the benign traffic, so results say
  little about real networks. Treat them as a pipeline check, not a security claim.
* **Small evaluation.** 13 test attack windows; one more or fewer detected moves recall by ~8 points.
* **Validation sweep is noisy.** Only 6 validation windows chose the alert quantile (0.999), and recall
  collapsed between quantile 0.98 (all windows, 112 incidents) and 0.99 on validation. There is no
  operating point that is both sensitive and quiet.
* **Training contamination.** The unlabelled training period contains attacks and benign look-alikes (the
  vuln scanner), so port scans look "normal" to the forest. This is the main reason for 0/4 port-scan recall.
* **Process disclosure.** The first run (easier generator, no confounders, no validation split) gave
  5/13 model recall and 13/13 baseline recall. The generator was then made harder and a validation split added;
  one feature-subset comparison was run on validation only and not adopted. Test results above come from
  a single scoring pass of the final configuration, but the design was changed after seeing earlier results.
* **Unused ideas that would likely help:** per-host baselines (host-relative z-scores), rate-of-change
  features, and training only on windows with no known incidents.
