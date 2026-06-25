from __future__ import annotations

import pandas as pd

def months_between(a, b) -> int:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    return (b.year - a.year) * 12 + (b.month - a.month)

def rolling_windows(train_month, mob_label: int = 6, oot_months: int = 3,
                    test_months: int = 3, origin_start: str = "2023-01-01") -> dict:
    M = pd.Timestamp(train_month).to_period("M").to_timestamp()
    oot_end   = M - pd.offsets.DateOffset(months=mob_label)
    oot_start = oot_end - pd.offsets.DateOffset(months=oot_months - 1)
    test_end  = oot_start - pd.offsets.DateOffset(months=1)
    test_start = test_end - pd.offsets.DateOffset(months=test_months - 1)
    train_end = test_start - pd.offsets.DateOffset(months=1)
    train_start = pd.Timestamp(origin_start)
    f = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
    return {"train_start": f(train_start), "train_end": f(train_end),
            "test_start": f(test_start), "test_end": f(test_end),
            "oot_start": f(oot_start), "oot_end": f(oot_end)}

def decide_training_action(month, champion_meta, monitoring_df, anchor,
                           refresh_months: int = 6, auc_floor: float = 0.70,
                           psi_retrain: float = 0.25):
    month = pd.Timestamp(month)
    anchor = pd.Timestamp(anchor)
    if month < anchor:
        return None, "before_anchor"
    if not champion_meta:
        return "bootstrap", "bootstrap_initial"

    last_train = pd.Timestamp(champion_meta.get("train_date", anchor))
    if months_between(last_train, month) >= refresh_months:
        return "refresh", "scheduled_cadence"

    if monitoring_df is not None and len(monitoring_df):
        mdf = monitoring_df.copy()
        if "month" in mdf.columns:
            mdf = mdf.sort_values("month")
        if "auc" in mdf.columns:
            lab = mdf[mdf["auc"].notna()]
            if len(lab) and float(lab.iloc[-1]["auc"]) < auc_floor:
                return "refresh", "trigger_auc_below_floor"
        if "psi" in mdf.columns:
            psi = mdf[mdf["psi"].notna()]
            if len(psi) and float(psi.iloc[-1]["psi"]) >= psi_retrain:
                return "refresh", "trigger_psi_breach"

    return None, "within_sla"

if __name__ == "__main__":
    import sys

    ok = True

    def check(name, cond):
        global ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print("rolling_windows(2024-06-01):")
    w = rolling_windows("2024-06-01")
    print("   ", w)
    check("oot_end == 2023-12-01 (M-6)", w["oot_end"] == "2023-12-01")
    check("oot is 3 months", months_between(w["oot_start"], w["oot_end"]) == 2)
    check("test ends before oot_start", pd.Timestamp(w["test_end"]) < pd.Timestamp(w["oot_start"]))
    check("train ends before test_start", pd.Timestamp(w["train_end"]) < pd.Timestamp(w["test_start"]))

    print("decide_training_action:")
    a, r = decide_training_action("2023-05-01", None, None, anchor="2024-06-01")
    check("before anchor -> None", a is None and r == "before_anchor")
    a, r = decide_training_action("2024-06-01", None, None, anchor="2024-06-01")
    check("at anchor, no champion -> bootstrap", a == "bootstrap")
    meta = {"train_date": "2024-06-01"}
    a, r = decide_training_action("2024-09-01", meta, None, anchor="2024-06-01")
    check("3 months later, healthy -> None", a is None and r == "within_sla")
    a, r = decide_training_action("2024-12-01", meta, None, anchor="2024-06-01")
    check("6 months later -> scheduled refresh", a == "refresh" and r == "scheduled_cadence")
    mon = pd.DataFrame({"month": ["2024-07-01", "2024-08-01"], "auc": [0.81, 0.66], "psi": [0.05, 0.08]})
    a, r = decide_training_action("2024-09-01", meta, mon, anchor="2024-06-01")
    check("AUC below floor -> refresh", a == "refresh" and r == "trigger_auc_below_floor")
    mon2 = pd.DataFrame({"month": ["2024-07-01", "2024-08-01"], "auc": [0.81, 0.80], "psi": [0.05, 0.31]})
    a, r = decide_training_action("2024-09-01", meta, mon2, anchor="2024-06-01")
    check("PSI breach -> refresh", a == "refresh" and r == "trigger_psi_breach")

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
