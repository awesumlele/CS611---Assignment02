import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import as02_config as config
from gold_io import read_gold

MIN_LABELLED = 30

def _ks(y, p):
    order = np.argsort(p); y = np.asarray(y)[order]
    cb = np.cumsum(y) / max(y.sum(), 1); cg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cb - cg)))

def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def _baseline_for(version, cache):
    if version in cache:
        return cache[version]
    candidates = [os.path.join(config.MODEL_BANK, str(version), "train_score_baseline.npy"),
                  os.path.join(config.PRODUCTION, "train_score_baseline.npy")]
    base = None
    for p in candidates:
        if os.path.exists(p):
            base = np.load(p); break
    cache[version] = base
    return base

def run_monitoring(run_date=None):
    parts = sorted(glob.glob(os.path.join(config.PREDICTIONS, "predictions_*.parquet")))
    if not parts:
        print("[monitor] no predictions yet - skipping")
        return pd.DataFrame()

    labels = read_gold(config.LABEL_STORE)[["loan_id", "label"]]
    cache, rows = {}, []
    for p in parts:
        preds = pd.read_parquet(p)
        month = pd.Timestamp(preds["orig_month"].iloc[0])
        version = preds["model_version"].iloc[0] if "model_version" in preds.columns else "unknown"
        merged = preds.merge(labels, on="loan_id", how="left")
        lab = merged.dropna(subset=["label"])
        baseline = _baseline_for(version, cache)
        row = {"month": month.strftime("%Y-%m-%d"), "model_version": version,
               "cohort_type": "labelled" if len(lab) >= MIN_LABELLED else "unlabelled",
               "n_scored": len(preds), "n_labelled": len(lab),
               "avg_score": round(float(preds["score"].mean()), 4),
               "pred_positive_rate": round(float(preds["prediction"].mean()), 4),
               "psi": round(psi(baseline, preds["score"].values), 4) if baseline is not None else None,
               "auc": None, "auc_pr": None, "ks": None, "actual_bad_rate": None}
        if len(lab) >= MIN_LABELLED and lab["label"].nunique() == 2:
            y = lab["label"].astype(int).values; s = lab["score"].values
            row["auc"] = round(float(roc_auc_score(y, s)), 4)
            row["auc_pr"] = round(float(average_precision_score(y, s)), 4)
            row["ks"] = round(_ks(y, s), 4)
            row["actual_bad_rate"] = round(float(y.mean()), 4)
        pd.DataFrame([row]).to_parquet(
            os.path.join(config.MONITORING, f"monitoring_{month.strftime('%Y_%m_%d')}.parquet"), index=False)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("month")
    summary.to_csv(os.path.join(config.MONITORING, "monitoring_summary.csv"), index=False)
    print(f"[monitor] {len(summary)} cohort(s) -> gold/model_monitoring")
    print(summary[["month", "model_version", "n_scored", "n_labelled", "auc", "psi"]].to_string(index=False))
    return summary

if __name__ == "__main__":
    run_monitoring()
