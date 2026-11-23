from pathlib import Path

# 1. Project Paths (works from notebooks/, src/, or Docker)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "payflow_transactions.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# 2. Target & Temporal Split Settings
TARGET_COL = "is_fraud"
TRAIN_DAYS = 120
VAL_DAYS = 30
TEST_DAYS = 30
RANDOM_STATE = 42

# 3. Feature Engineering Constants
FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 0.92,
    "GBP": 0.79,
    "SAR": 3.75,
    "EGP": 48.5,
}
CHARGEBACK_LAG_DAYS = 7
BAYESIAN_SMOOTHING_M = 20

# 4. Model Feature Schema
NUMERIC_COLS: list[str] = [
    "account_age_days",
    "failed_attempts_1h",
    "hour_of_day",
    "user_tx_count_1h",
    "user_tx_count_24h",
    "merchant_fraud_rate_7d_lag",
]

LOG_NUMERIC_COLS: list[str] = [
    "amount_usd",
    "distance_from_home_km",
    "time_since_last_tx_sec",
    "amount_to_user_avg_ratio",
]

BINARY_COLS: list[str] = [
    "kyc_verified",
    "is_new_device",
    "is_vpn_or_proxy",
    "is_international",
    "is_night",
    "ip_country_mismatch",
]

CATEGORICAL_COLS: list[str] = [
    "currency",
    "merchant_category",
    "payment_method",
    "entry_mode",
    "user_home_country",
    "device_type",
    "ip_country",
]

FEATURE_COLS: list[str] = (
    NUMERIC_COLS + LOG_NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS
)

DROP_COLS: list[str] = [
    "timestamp",
    "transaction_id",
    "user_id",
    "merchant_id",
    "amount",
    "user_avg_amount_usd",
]