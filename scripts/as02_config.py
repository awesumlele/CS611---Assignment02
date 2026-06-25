import os

ROOT = os.environ.get("PIPELINE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA_DIR   = os.path.join(ROOT, "data")
DATAMART   = os.path.join(ROOT, "datamart")
BRONZE     = os.path.join(DATAMART, "bronze")
SILVER     = os.path.join(DATAMART, "silver")
GOLD       = os.path.join(DATAMART, "gold")
FEATURE_STORE = os.path.join(GOLD, "feature_store")
LABEL_STORE   = os.path.join(GOLD, "label_store")
PREDICTIONS   = os.path.join(GOLD, "model_predictions")
MONITORING    = os.path.join(GOLD, "model_monitoring")
VIZ_DIR       = os.path.join(GOLD, "monitoring_viz")
MODEL_BANK    = os.path.join(ROOT, "model_bank")

PRODUCTION    = os.path.join(MODEL_BANK, "production")
REGISTRY      = os.path.join(MODEL_BANK, "model_registry.csv")
LEADERBOARD   = os.path.join(MODEL_BANK, "leaderboard.json")

for _p in (PREDICTIONS, MONITORING, VIZ_DIR, MODEL_BANK, PRODUCTION):
    os.makedirs(_p, exist_ok=True)

SNAPSHOT_START = "2023-01-01"
SNAPSHOT_END   = "2024-12-01"

DPD_THRESHOLD = 30
MOB_LABEL     = 6
LABEL_DEF     = f"{DPD_THRESHOLD}dpd_{MOB_LABEL}mob"

GOV_ANCHOR     = "2024-06-01"
INITIAL_MODEL  = "logistic_regression"
REFRESH_MONTHS = 6
OOT_MONTHS     = 3
TEST_MONTHS    = 3

AUC_FLOOR          = 0.70
PSI_WATCH          = 0.10
PSI_RETRAIN        = 0.25
PROMOTION_GATE_AUC = 0.60

RANDOM_STATE = 42
MODEL_SELECTION_METRIC = "auc"

DROP_COLS = ["Customer_ID", "loan_id", "snapshot_date", "feat_snapshot",
             "label", "label_def", "orig_month", "Type_of_Loan",
             "model_name", "model_version", "score", "prediction", "scored_at"]

CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
