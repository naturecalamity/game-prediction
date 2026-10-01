import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier

# 1. Mock Data Construction
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame(
    {
        "match_id": range(n_samples),
        "team_id": np.random.choice([101, 102, 103, 104], size=n_samples),
        "match_date": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
        "goals_scored": np.random.poisson(lam=1.5, size=n_samples),
        "goals_conceded": np.random.poisson(lam=1.1, size=n_samples),
        "xg_created": np.random.uniform(0.5, 3.0, size=n_samples),
        "possession": np.random.uniform(40, 65, size=n_samples),
        "win": np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45]),
    }
).sort_values(["team_id", "match_date"])


# 2. Separate Feature Engineering (Short vs. Long Term)
def engineer_timescale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("team_id")

    # Short-Term Inputs (3–5 match windows)
    df["st_goals_scored_avg_5"] = grouped["goals_scored"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["st_goals_conceded_avg_5"] = grouped["goals_conceded"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["st_xg_created_avg_3"] = grouped["xg_created"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df["st_win_rate_5"] = grouped["win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Long-Term Inputs (30+ match windows)
    df["lt_goals_scored_avg_30"] = grouped["goals_scored"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=5).mean()
    )
    df["lt_goals_conceded_avg_30"] = grouped["goals_conceded"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=5).mean()
    )
    df["lt_xg_created_avg_30"] = grouped["xg_created"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=5).mean()
    )
    df["lt_possession_avg_30"] = grouped["possession"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=5).mean()
    )

    # Ratio interaction feature
    df["ratio_xg_short_vs_long"] = (df["st_xg_created_avg_3"] + 1e-5) / (
        df["lt_xg_created_avg_30"] + 1e-5
    )

    return df


df_featured = engineer_timescale_features(data).dropna()

SHORT_TERM_FEATURES = [
    "st_goals_scored_avg_5",
    "st_goals_conceded_avg_5",
    "st_xg_created_avg_3",
    "st_win_rate_5",
]

LONG_TERM_FEATURES = [
    "lt_goals_scored_avg_30",
    "lt_goals_conceded_avg_30",
    "lt_xg_created_avg_30",
    "lt_possession_avg_30",
]

INTERACTION_FEATURES = ["ratio_xg_short_vs_long"]
ALL_FEATURES = SHORT_TERM_FEATURES + LONG_TERM_FEATURES + INTERACTION_FEATURES
TARGET = "win"

# 3. Time-Based Data Split
split_idx = int(len(df_featured) * 0.8)
X_train = df_featured[ALL_FEATURES].iloc[:split_idx]
y_train = df_featured[TARGET].iloc[:split_idx]
X_val = df_featured[ALL_FEATURES].iloc[split_idx:]
y_val = df_featured[TARGET].iloc[split_idx:]

# 4. Train XGBoost Model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=30,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)

# 5. Evaluate Metrics & Print Feature Group Importances
preds = model.predict_proba(X_val)[:, 1]
print(f"Validation Log Loss: {log_loss(y_val, preds):.4f}")
print(f"Validation ROC-AUC:  {roc_auc_score(y_val, preds):.4f}\n")

importance_df = pd.DataFrame(
    {"feature": ALL_FEATURES, "importance": model.feature_importances_}
)

importance_df["type"] = importance_df["feature"].apply(
    lambda x: (
        "Short-Term"
        if x in SHORT_TERM_FEATURES
        else ("Long-Term" if x in LONG_TERM_FEATURES else "Interaction")
    )
)

print("=== Feature Importance Split by Input Group ===")
print(
    importance_df.groupby("type")["importance"]
    .sum()
    .reset_index()
    .to_string(index=False)
)
