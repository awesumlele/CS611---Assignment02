import os
import glob
import json
import pickle
import re
import datetime as dt

import numpy as np
import pandas as pd

import as02_config as config
from gold_io import read_gold
from feature_prep import prepare_features

def _load_production():
    prod = config.PRODUCTION
    mpath = os.path.join(prod, "model.pkl")
    meta_path = os.path.join(prod, "production_metadata.json")
    if not (os.path.exists(mpath) and os.path.exists(meta_path)):
        return None, None
    with open(mpath, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

def _prepared_features():
    feats = read_gold(config.FEATURE_STORE).rename(columns={"snapshot_date": "feat_snapshot"})
    if feats.empty:
        return feats
    feats["feat_snapshot"] = pd.to_datetime(feats["feat_snapshot"])
    feats = feats.sort_values(["Customer_ID", "feat_snapshot"]).drop_duplicates(
        ["Customer_ID", "feat_snapshot"], keep="last")
    return prepare_features(feats)

def _read_silver_loans():
    parts = sorted(glob.glob(os.path.join(config.SILVER, "silver_loans_*.parquet")))
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _already_scored_months():
    done = set()
    for p in glob.glob(os.path.join(config.PREDICTIONS, "predictions_*.parquet")):
        m = re.search(r"predictions_(\d{4})_(\d{2})_(\d{2})\.parquet$", os.path.basename(p))
        if m:
            done.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return done

def run_inference(run_date=None):
    model, meta = _load_production()
    if model is None:
        print("[inference] no champion in production yet (bootstrap pending) - skipping")
        return 0
    feat_cols = meta["feature_columns"]

    feats = _prepared_features()
    loans = _read_silver_loans()
    if feats.empty or loans.empty:
        print("[inference] gold features / silver loans not found - run build_datamart first")
        return 0
    loans["snapshot_date"] = pd.to_datetime(loans["snapshot_date"])

    apps = loans[loans["mob"] == 0][["loan_id", "Customer_ID", "snapshot_date"]].copy()
    apps = apps.rename(columns={"snapshot_date": "orig_month"})

    apps = apps.sort_values("orig_month")
    feats = feats.sort_values("feat_snapshot")
    scored = pd.merge_asof(apps, feats, by="Customer_ID",
                           left_on="orig_month", right_on="feat_snapshot",
                           direction="backward")
    for c in feat_cols:
        if c not in scored.columns:
            scored[c] = np.nan

    scored["score"] = np.round(model.predict_proba(scored[feat_cols])[:, 1], 6)
    scored["prediction"] = (scored["score"] >= meta["decision_threshold"]).astype(int)
    scored["model_name"] = meta["model_name"]
    scored["model_version"] = meta["version"]
    scored["scored_at"] = dt.datetime.now().isoformat(timespec="seconds")

    out_cols = ["loan_id", "Customer_ID", "orig_month", "score", "prediction",
                "model_name", "model_version", "scored_at"]

    done = _already_scored_months()
    total, written = 0, 0
    for month, grp in scored.groupby("orig_month"):
        ds = pd.Timestamp(month).strftime("%Y-%m-%d")
        if ds in done:
            continue
        fname = f"predictions_{pd.Timestamp(month).strftime('%Y_%m_%d')}.parquet"
        grp[out_cols].to_parquet(os.path.join(config.PREDICTIONS, fname), index=False)
        total += len(grp); written += 1
    print(f"[inference] champion {meta['version']} ({meta['model_name']}) "
          f"scored {total} applications across {written} new cohort(s); "
          f"{len(done)} cohort(s) already existed")
    return total

if __name__ == "__main__":
    run_inference()
