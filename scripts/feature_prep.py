import re
import numpy as np
import pandas as pd

import as02_config as config

_DIRTY_NUMERIC = ["Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
                  "Amount_invested_monthly", "Monthly_Balance", "Num_Credit_Inquiries",
                  "Outstanding_Debt", "Annual_Income", "Monthly_Inhand_Salary",
                  "Total_EMI_per_month", "Credit_Utilization_Ratio", "Age",
                  "Interest_Rate", "Delay_from_due_date", "Num_Bank_Accounts", "Num_Credit_Card"]

_SENTINEL_CLIP = {
    "Num_Bank_Accounts": (0, 20), "Num_Credit_Card": (0, 15), "Num_of_Loan": (0, 20),
    "Interest_Rate": (0, 50), "Num_of_Delayed_Payment": (0, 60), "Age": (18, 100),
    "Num_Credit_Inquiries": (0, 50),
}

def _to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace("_", "", regex=False).str.strip()
         .replace({"": np.nan, "nan": np.nan, "NM": np.nan, "!@9#%8": np.nan}),
        errors="coerce")

def _credit_history_months(s):
    def conv(v):
        if not isinstance(v, str):
            return np.nan
        y = re.search(r"(\d+)\s*Year", v); m = re.search(r"(\d+)\s*Month", v)
        if y is None and m is None:
            return np.nan
        return (int(y.group(1)) if y else 0) * 12 + (int(m.group(1)) if m else 0)
    return s.apply(conv)

def prepare_features(df):
    df = df.copy()

    if "Credit_History_Age" in df.columns:
        df["credit_history_months"] = _credit_history_months(df["Credit_History_Age"])
        df = df.drop(columns=["Credit_History_Age"])
    if "Type_of_Loan" in df.columns:
        def _count_types(v):
            if not isinstance(v, str):
                return 0
            parts = [t.strip() for t in v.split(",")]
            return len([t for t in parts if t and t.lower() != "nan"])
        df["num_loan_types"] = df["Type_of_Loan"].apply(_count_types)

    for c in _DIRTY_NUMERIC:
        if c in df.columns:
            df[c] = _to_num(df[c])

    for c, (lo, hi) in _SENTINEL_CLIP.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)

    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(df.columns):
        df["emi_to_salary"] = (df["Total_EMI_per_month"] /
                               df["Monthly_Inhand_Salary"].replace(0, np.nan)).clip(0, 5)
    if {"Outstanding_Debt", "Annual_Income"}.issubset(df.columns):
        df["debt_to_income"] = (df["Outstanding_Debt"] /
                                df["Annual_Income"].replace(0, np.nan)).clip(0, 5)

    for c in config.CATEGORICAL:
        if c in df.columns:
            df[c] = (df[c].astype(str).str.strip()
                     .replace({"_": "Unknown", "": "Unknown", "nan": "Unknown",
                               "!@9#%8": "Unknown", "NM": "Unknown"}))
    return df

def feature_columns(df):
    cat = [c for c in config.CATEGORICAL if c in df.columns]
    num = [c for c in df.columns
           if c not in config.DROP_COLS + cat and pd.api.types.is_numeric_dtype(df[c])]
    return num, cat
