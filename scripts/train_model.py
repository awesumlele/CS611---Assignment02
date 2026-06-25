import os
import json
import pickle
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import as02_config as config
import governance as gov
from gold_io import read_gold
from feature_prep import prepare_features, feature_columns

def build_modelling_table():
    feats = read_gold(config.FEATURE_STORE)
    labels = read_gold(config.LABEL_STORE)
    if feats.empty or labels.empty:
        return pd.DataFrame()

    feats = feats.rename(columns={"snapshot_date": "feat_snapshot"})
    feats["feat_snapshot"] = pd.to_datetime(feats["feat_snapshot"])
    feats = feats.sort_values("feat_snapshot").drop_duplicates("Customer_ID", keep="last")
    feats = prepare_features(feats)

    labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"])
    labels["orig_month"] = labels["snapshot_date"] - pd.DateOffset(months=config.MOB_LABEL)

    df = labels.merge(feats, on="Customer_ID", how="inner")
    if df.empty:
        return df
    df = df[df["feat_snapshot"] <= df["orig_month"]]
    return df

def _window_split(df, w):
    om = df["orig_month"]
    tr  = df[(om >= w["train_start"]) & (om <= w["train_end"])]
    te  = df[(om >= w["test_start"])  & (om <= w["test_end"])]
    oot = df[(om >= w["oot_start"])   & (om <= w["oot_end"])]
    return tr, te, oot

def _ks(y, p):
    order = np.argsort(p); y = np.asarray(y)[order]
    cb = np.cumsum(y) / max(y.sum(), 1); cg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(cb - cg)))

def _metrics(y, p):
    if len(np.unique(y)) < 2:
        return {"auc": None, "auc_pr": None, "gini": None, "ks": None,
                "bad_rate": round(float(np.mean(y)), 4), "n": int(len(y))}
    auc = roc_auc_score(y, p)
    return {"auc": round(float(auc), 4), "auc_pr": round(float(average_precision_score(y, p)), 4),
            "gini": round(float(2 * auc - 1), 4), "ks": round(_ks(y, p), 4),
            "bad_rate": round(float(np.mean(y)), 4), "n": int(len(y))}

def _threshold_for_recall(y, p, target=0.75):
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return 0.5
    return float(thr[ok[np.argmax(prec[:-1][ok])]])

def _make_pipe(model, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=20))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", model)])

def _candidates(spw):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=config.RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 reg_lambda=1.0, gamma=0.1, eval_metric="logloss",
                                 scale_pos_weight=spw, random_state=config.RANDOM_STATE),
        "lightgbm": LGBMClassifier(n_estimators=400, max_depth=4, num_leaves=31, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                                   reg_lambda=1.0, scale_pos_weight=spw,
                                   random_state=config.RANDOM_STATE, verbose=-1),
    }

def _load_champion_meta():
    p = os.path.join(config.PRODUCTION, "production_metadata.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception as e:
            print(f"[train] WARN could not read production metadata: {e}")
    return None

def _load_monitoring():
    p = os.path.join(config.MONITORING, "monitoring_summary.csv")
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def _append_registry(row):
    df = pd.DataFrame([row])
    if os.path.exists(config.REGISTRY):
        prev = pd.read_csv(config.REGISTRY)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=["model_version"], keep="last")
    df.to_csv(config.REGISTRY, index=False)

def _select(results):
    def score(n):
        oot = results[n]["oot"].get("auc")
        return oot if oot is not None else (results[n]["test"].get("auc") or 0.0)
    return max(results, key=score)

def run_training(run_date=None):
    month = (run_date or dt.date.today().isoformat())[:10]
    champion_meta = _load_champion_meta()
    monitoring_df = _load_monitoring()

    action, reason = gov.decide_training_action(
        month, champion_meta, monitoring_df, anchor=config.GOV_ANCHOR,
        refresh_months=config.REFRESH_MONTHS, auc_floor=config.AUC_FLOOR,
        psi_retrain=config.PSI_RETRAIN)
    print(f"[train] {month}: action={action} reason={reason}")
    if action is None:
        return {"action": None, "reason": reason}

    df = build_modelling_table()
    if df.empty:
        print("[train] modelling table empty (datamart not built yet?) - skipping")
        return {"action": None, "reason": "no_data"}

    w = gov.rolling_windows(month, mob_label=config.MOB_LABEL,
                            oot_months=config.OOT_MONTHS, test_months=config.TEST_MONTHS,
                            origin_start=config.SNAPSHOT_START)
    tr, te, oot = _window_split(df, w)
    num, cat = feature_columns(df)
    feat_cols = num + cat
    print(f"[train] windows train<= {w['train_end']} | test {w['test_start']}..{w['test_end']} | "
          f"oot {w['oot_start']}..{w['oot_end']}")
    print(f"[train] features={len(feat_cols)} train={len(tr)} test={len(te)} oot={len(oot)}")
    if len(tr) == 0 or tr["label"].nunique() < 2:
        print("[train] insufficient / single-class training window - skipping")
        return {"action": None, "reason": "insufficient_train"}

    y_tr = tr["label"].astype(int).values
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    pool = _candidates(spw)
    names = [config.INITIAL_MODEL] if action == "bootstrap" else list(pool.keys())
    candidates = {k: pool[k] for k in names}

    results, pipes = {}, {}
    for name, model in candidates.items():
        pipe = _make_pipe(model, num, cat)
        pipe.fit(tr[feat_cols], y_tr)
        m = {"train": _metrics(y_tr, pipe.predict_proba(tr[feat_cols])[:, 1])}
        m["test"] = _metrics(te["label"].astype(int).values, pipe.predict_proba(te[feat_cols])[:, 1]) \
            if len(te) else {"auc": None}
        m["oot"] = _metrics(oot["label"].astype(int).values, pipe.predict_proba(oot[feat_cols])[:, 1]) \
            if len(oot) else {"auc": None}
        results[name] = m; pipes[name] = pipe
        print(f"[train] {name:20s} train_auc={m['train']['auc']} "
              f"test_auc={m['test'].get('auc')} oot_auc={m['oot'].get('auc')}")

    best = _select(results)
    sel_auc = results[best]["oot"].get("auc") or results[best]["test"].get("auc")
    promoted = bool(sel_auc is not None and sel_auc >= config.PROMOTION_GATE_AUC)
    eval_split = te if len(te) else tr
    threshold = round(_threshold_for_recall(eval_split["label"].astype(int).values,
                                            pipes[best].predict_proba(eval_split[feat_cols])[:, 1], 0.75), 4)
    version = "credit_model_" + month.replace("-", "_")
    print(f"[train] action={action} best={best} oot/test_auc={sel_auc} promoted={promoted} "
          f"threshold={threshold} version={version}")

    vdir = os.path.join(config.MODEL_BANK, version); os.makedirs(vdir, exist_ok=True)
    with open(os.path.join(vdir, "model.pkl"), "wb") as f: pickle.dump(pipes[best], f)
    with open(os.path.join(vdir, "metrics.json"), "w") as f: json.dump(results[best], f, indent=2)
    train_baseline = pipes[best].predict_proba(tr[feat_cols])[:, 1]
    np.save(os.path.join(vdir, "train_score_baseline.npy"), train_baseline)

    os.makedirs(config.PRODUCTION, exist_ok=True)
    with open(os.path.join(config.PRODUCTION, "model.pkl"), "wb") as f: pickle.dump(pipes[best], f)
    np.save(os.path.join(config.PRODUCTION, "train_score_baseline.npy"), train_baseline)
    meta = {"model_name": best, "version": version, "train_date": month,
            "action": action, "reason": reason, "promoted": promoted,
            "selection_metric": "oot_auc", "decision_threshold": threshold,
            "label_def": config.LABEL_DEF, "feature_columns": feat_cols, "categorical": cat,
            "windows": w, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "metrics": results[best]}
    with open(os.path.join(config.PRODUCTION, "production_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(config.LEADERBOARD, "w") as f:
        json.dump({"month": month, "action": action, "results": results,
                   "selected": best, "threshold": threshold}, f, indent=2)

    _append_registry({
        "train_date": month, "model_version": version, "action": action, "reason": reason,
        "champion_model": best, "candidates": "|".join(candidates.keys()),
        "promoted": promoted, "decision_threshold": threshold,
        "auc_train": results[best]["train"].get("auc"),
        "auc_test": results[best]["test"].get("auc"),
        "auc_oot": results[best]["oot"].get("auc"),
        "n_train": len(tr), "n_test": len(te), "n_oot": len(oot),
    })
    print(f"[train] champion '{best}' promoted as {version}; registry -> {config.REGISTRY}")
    return meta

def train():
    return run_training()

if __name__ == "__main__":
    run_training()
