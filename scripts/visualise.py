import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import as02_config as config

NAVY, BLUE, GREY, AMBER, RED, GREEN = "#1f2d4d", "#2e6fb0", "#9aa3b2", "#e0a300", "#c0392b", "#2e8b57"
ERA = [BLUE, GREEN, AMBER, "#6b4c9a"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

def _summary():
    df = pd.read_csv(os.path.join(config.MONITORING, "monitoring_summary.csv"))
    df["month"] = pd.to_datetime(df["month"])
    return df.sort_values("month")

def _registry():
    if os.path.exists(config.REGISTRY):
        try:
            r = pd.read_csv(config.REGISTRY)
            r["train_date"] = pd.to_datetime(r["train_date"])
            return r.sort_values("train_date")
        except Exception:
            pass
    return pd.DataFrame()

def _eras(ax, df):
    if "model_version" not in df.columns:
        return
    for i, (ver, g) in enumerate(df.groupby("model_version")):
        lo, hi = g["month"].min(), g["month"].max()
        ax.axvspan(lo, hi, color=ERA[i % len(ERA)], alpha=0.06)

def _markers(ax, reg):
    for _, r in reg.iterrows():
        ax.axvline(r["train_date"], color=NAVY, ls=":", lw=1.2, alpha=0.8)
        tag = "bootstrap" if str(r.get("action")) == "bootstrap" else "refresh"
        ax.text(r["train_date"], ax.get_ylim()[0], f" {tag}", color=NAVY, fontsize=7.5,
                rotation=90, va="bottom", ha="right")

def perf_over_time(df, reg):
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    lab = df.dropna(subset=["auc"])
    _eras(ax, df)
    ax.plot(lab.month, lab.auc, "o-", color=BLUE, lw=2.2, ms=5, label="AUC-ROC (matured cohorts)")
    ax.axhline(config.AUC_FLOOR, color=RED, ls=":", lw=1.5, label=f"Performance floor (AUC {config.AUC_FLOOR})")
    if len(reg):
        _markers(ax, reg)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("AUC-ROC")
    ax.set_title("Model performance across time", fontweight="bold", color=NAVY)
    ax.legend(fontsize=8.5, loc="lower left"); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "perf_over_time.png")); plt.close(fig)

def psi_over_time(df, reg):
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    d = df.dropna(subset=["psi"])
    colors = [GREEN if v < config.PSI_WATCH else (AMBER if v < config.PSI_RETRAIN else RED) for v in d.psi]
    ax.bar(d.month, d.psi, width=20, color=colors)
    ax.axhline(config.PSI_WATCH, color=AMBER, ls="--", lw=1.3, label=f"Watch ({config.PSI_WATCH})")
    ax.axhline(config.PSI_RETRAIN, color=RED, ls="--", lw=1.3, label=f"Retrain ({config.PSI_RETRAIN})")
    if len(reg):
        _markers(ax, reg)
    ax.set_ylabel("PSI (score distribution)")
    ax.set_title("Population Stability Index across time", fontweight="bold", color=NAVY)
    ax.legend(fontsize=8.5, loc="upper left"); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "psi_over_time.png")); plt.close(fig)

def score_drift(df):
    base_path = os.path.join(config.PRODUCTION, "train_score_baseline.npy")
    if not os.path.exists(base_path):
        return
    baseline = np.load(base_path)
    parts = sorted(glob.glob(os.path.join(config.PREDICTIONS, "predictions_*.parquet")))
    latest = pd.read_parquet(parts[-1]) if parts else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8.4, 3.7)); bins = np.linspace(0, 1, 26)
    ax.hist(baseline, bins=bins, density=True, alpha=0.55, color=BLUE, label="Training baseline")
    if not latest.empty:
        ax.hist(latest.score, bins=bins, density=True, alpha=0.55, color=AMBER,
                label=f"Latest cohort ({pd.Timestamp(latest['orig_month'].iloc[0]).date()})")
    ax.set_xlabel("Predicted default probability"); ax.set_ylabel("Density")
    ax.set_title("Score distribution drift", fontweight="bold", color=NAVY)
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "score_drift.png")); plt.close(fig)

def pred_vs_actual(df):
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    ax.plot(df.month, df.pred_positive_rate, "o-", color=BLUE, lw=2, ms=4, label="Predicted positive rate")
    lab = df.dropna(subset=["actual_bad_rate"])
    ax.plot(lab.month, lab.actual_bad_rate, "s-", color=NAVY, lw=2, ms=4, label="Actual bad rate (matured)")
    ax.set_ylabel("Rate"); ax.set_title("Predicted vs actual default rate", fontweight="bold", color=NAVY)
    ax.legend(fontsize=9); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "pred_vs_actual.png")); plt.close(fig)

def _auc_of(split_metrics):
    oot = split_metrics.get("oot", {}).get("auc")
    return oot if oot is not None else split_metrics.get("test", {}).get("auc")

def leaderboard():
    if not os.path.exists(config.LEADERBOARD):
        return
    with open(config.LEADERBOARD) as f:
        lb = json.load(f)
    res, sel = lb["results"], lb["selected"]
    names = list(res.keys())
    aucs = [(_auc_of(res[n]) or 0.0) for n in names]
    colors = [GREEN if n == sel else BLUE for n in names]
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    bars = ax.bar([n.replace("_", "\n") for n in names], aucs, color=colors, width=0.55)
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width()/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(config.PROMOTION_GATE_AUC, color=RED, ls=":", lw=1.4,
               label=f"Promotion gate ({config.PROMOTION_GATE_AUC})")
    ax.set_ylim(0.5, max(aucs + [0.7]) + 0.05); ax.set_ylabel("OOT AUC-ROC")
    ax.set_title("Latest bake-off candidate AUC (champion = green)", fontweight="bold", color=NAVY)
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(config.VIZ_DIR, "leaderboard.png")); plt.close(fig)

def run_visualise(run_date=None):
    summary_csv = os.path.join(config.MONITORING, "monitoring_summary.csv")
    if not os.path.exists(summary_csv):
        print("[visualise] no monitoring summary yet - skipping")
        return
    df = _summary()
    reg = _registry()
    perf_over_time(df, reg); psi_over_time(df, reg); score_drift(df); pred_vs_actual(df); leaderboard()
    print(f"[visualise] charts -> {config.VIZ_DIR}")

if __name__ == "__main__":
    run_visualise()
