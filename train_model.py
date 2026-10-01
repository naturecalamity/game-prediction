import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_and_preprocess_data(csv_path):
    """Loads raw matches CSV and generates short-term (5-game) and long-term (30-game) features."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Run featch_apisports.py first!"
        )

    df = pd.read_csv(csv_path)

    # 1. Target Label Creation (1 = Home Win, 0 = Draw/Away Win)
    # Adjust column names depending on raw API response structure
    if "goals.home" in df.columns:
        # Soccer format
        df["target"] = (df["goals.home"] > df["goals.away"]).astype(int)
        df["home_score"] = df["goals.home"]
        df["away_score"] = df["goals.away"]
    elif "scores.home.total" in df.columns:
        # Basketball format
        df["target"] = (df["scores.home.total"] > df["scores.away.total"]).astype(int)
        df["home_score"] = df["scores.home.total"]
        df["away_score"] = df["scores.away.total"]
    else:
        # Fallback dummy check if raw columns differ
        raise KeyError(
            "Could not detect score columns in CSV. Inspect your raw CSV headers."
        )

    # Clean missing targets
    df = df.dropna(subset=["target"]).copy()

    # 2. Feature Engineering: Short-Term (ST) vs Long-Term (LT) Rolling Features
    # Short-Term Form (5-game moving average)
    df["st_home_goals"] = df["home_score"].rolling(window=5, min_periods=1).mean()
    df["st_away_goals"] = df["away_score"].rolling(window=5, min_periods=1).mean()

    # Long-Term Baseline (30-game moving average)
    df["lt_home_goals"] = df["home_score"].rolling(window=30, min_periods=1).mean()
    df["lt_away_goals"] = df["away_score"].rolling(window=30, min_periods=1).mean()

    # Interactive Ratios (ST Form relative to LT Baseline)
    df["home_form_ratio"] = df["st_home_goals"] / (df["lt_home_goals"] + 1e-5)
    df["away_form_ratio"] = df["st_away_goals"] / (df["lt_away_goals"] + 1e-5)

    feature_cols = [
        "st_home_goals",
        "st_away_goals",
        "lt_home_goals",
        "lt_away_goals",
        "home_form_ratio",
        "away_form_ratio",
    ]

    X = df[feature_cols].fillna(0)
    y = df["target"]

    return X, y, feature_cols


def train_xgboost():
    print("--- STARTING MODEL TRAINING ---")

    # Load Soccer Data
    csv_file = "data/raw_soccer_matches.csv"
    X, y, feature_names = load_and_preprocess_data(csv_file)

    print(f"Dataset shape: {X.shape[0]} matches, {X.shape[1]} features")

    # Split into 80% Train, 20% Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize XGBoost Classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )

    # Train Model
    print("Training XGBoost Model...")
    model.fit(X_train, y_train)

    # Evaluate Model
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs)

    print("\n--- MODEL EVALUATION RESULTS ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Log Loss: {loss:.4f}")

    # Feature Importance Breakdown
    print("\n--- FEATURE IMPORTANCE BREAKDOWN ---")
    importance = model.feature_importances_
    for name, imp in zip(feature_names, importance):
        print(f"{name:20s}: {imp:.4f}")

    # Save trained model to disk
    os.makedirs("models", exist_ok=True)
    model.save_model("models/xgboost_soccer.json")
    print("\n--> SUCCESS! Saved trained model to models/xgboost_soccer.json")


if __name__ == "__main__":
    train_xgboost()
