import pandas as pd

from src.config import (
     BAYESIAN_SMOOTHING_M,
     CHARGEBACK_LAG_DAYS,
     DROP_COLS,
     FX_TO_USD,
     RAW_DATA_PATH,
     TARGET_COL,
     TEST_DAYS,
     TRAIN_DAYS,
     VAL_DAYS,
)

def add_base_features(df: pd.DataFrame) -> pd.DataFrame:

     df = df.copy()
     df["timestamp"] = pd.to_datetime(df["timestamp"])
     df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

     df["amount_usd"] = df["amount"] / df["currency"].map(FX_TO_USD).round(2)
     df["hour_of_day"] = df["timestamp"].dt.hour
     df["is_night"] = df["timestamp"].dt.hour.between(1, 5).astype(int)
     df["ip_country_mismatch"] = (
          (df["ip_country"].notna()) & (df["ip_country"] != df["user_home_country"])
     ).astype(int)

     return df

def add_user_features(df: pd.DataFrame) -> pd.DataFrame:

     df = df.sort_values(["user_id", "timestamp", "transaction_id"]).reset_index(drop=True)

     for col_name, window in [("user_tx_count_1h", "1h"), ("user_tx_count_24h", "24h")]:
          df[col_name] = (
               df.groupby("user_id", sort=False)
               .rolling(window, on="timestamp", closed="left")["amount_usd"]
               .count()
               .fillna(0)
               .astype(int)
               .to_numpy()
          )

     df["time_since_last_tx_sec"] = (
          df.groupby("user_id", sort=False)["timestamp"].diff().dt.total_seconds()
     )

     user_avg_amount = (
          df.groupby("user_id", sort=False)["amount_usd"]
          .transform(lambda s: s.shift(1).expanding().mean())
     )
     df["amount_to_user_avg_ratio"] = df["amount_usd"] / user_avg_amount

     return df


def get_split_boundaries(
          df: pd.DataFrame,
          train_days: int = TRAIN_DAYS,
          val_days: int = VAL_DAYS,
          test_days: int = TEST_DAYS
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:

     start = df["timestamp"].min().normalize()
     train_end = start + pd.Timedelta(days=train_days)
     val_end = train_end + pd.Timedelta(days=val_days)
     test_end = val_end + pd.Timedelta(days=test_days)
     return train_end, val_end, test_end

def build_and_split_data(
          df: pd.DataFrame,
          train_days: int = TRAIN_DAYS,
          val_days: int = VAL_DAYS,
          test_days: int = TEST_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

     df = add_base_features(df)
     train_end, val_end, test_end = get_split_boundaries(
          df, train_days=train_days, val_days=val_days, test_days=test_days
     )
     df = add_user_features(df)

     train_df = df[df["timestamp"] < train_end].copy()
     val_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)].copy()
     test_df = df[(df["timestamp"] >= val_end) & (df["timestamp"] < test_end)].copy()

     def _split_xy(part: pd.DataFrame) -> pd.DataFrame:
          X = part.drop(columns=DROP_COLS + [TARGET_COL], errors="ignore")
          y = part[TARGET_COL].copy()
          return X, y

     X_train, y_train = _split_xy(train_df)
     X_val, y_val = _split_xy(val_df)
     X_test, y_test = _split_xy(test_df)

     return X_train, y_train, X_val, y_val, X_test, y_test
