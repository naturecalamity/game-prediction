import os
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

# 1. Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_soccer.json"
HISTORICAL_CSV = BASE_DIR / "data" / "raw_soccer_matches.csv"


# -------------------------------------------------------------
# 1. LOAD MODEL
# -------------------------------------------------------------
def load_soccer_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first!"
        )
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


# -------------------------------------------------------------
# 2. COMPUTE TEAM FEATURES FROM HISTORICAL CSV
# -------------------------------------------------------------
def get_team_features(home_team, away_team):
    """Calculates rolling averages for the specified home and away teams."""
    if not os.path.exists(HISTORICAL_CSV):
        raise FileNotFoundError(
            f"Historical CSV not found at {HISTORICAL_CSV}. Run featch_apisports.py first!"
        )

    df = pd.read_csv(HISTORICAL_CSV)
    df["home_score"] = df.get("goals.home", 0)
    df["away_score"] = df.get("goals.away", 0)

    # Calculate global default averages as fallbacks for unknown teams
    global_st_home = df["home_score"].tail(5).mean()
    global_st_away = df["away_score"].tail(5).mean()
    global_lt_home = df["home_score"].tail(30).mean()
    global_lt_away = df["away_score"].tail(30).mean()

    # Filter home team history
    home_matches = df[
        df["teams.home.name"].str.contains(home_team, case=False, na=False)
    ]
    if not home_matches.empty:
        st_home = home_matches["home_score"].tail(5).mean()
        lt_home = home_matches["home_score"].tail(30).mean()
    else:
        print(f"⚠️ Home team '{home_team}' not found in history. Using league average.")
        st_home, lt_home = global_st_home, global_lt_home

    # Filter away team history
    away_matches = df[
        df["teams.away.name"].str.contains(away_team, case=False, na=False)
    ]
    if not away_matches.empty:
        st_away = away_matches["away_score"].tail(5).mean()
        lt_away = away_matches["away_score"].tail(30).mean()
    else:
        print(f"⚠️ Away team '{away_team}' not found in history. Using league average.")
        st_away, lt_away = global_st_away, global_lt_away

    # Feature calculation
    home_form_ratio = st_home / (lt_home + 1e-5)
    away_form_ratio = st_away / (lt_away + 1e-5)

    features = pd.DataFrame(
        [
            {
                "st_home_goals": st_home,
                "st_away_goals": st_away,
                "lt_home_goals": lt_home,
                "lt_away_goals": lt_away,
                "home_form_ratio": home_form_ratio,
                "away_form_ratio": away_form_ratio,
            }
        ]
    )

    return features


# -------------------------------------------------------------
# 3. INTERACTIVE PREDICTION LOOP
# -------------------------------------------------------------
def run_user_prediction():
    model = load_soccer_model()

    print("=" * 65)
    print("      CUSTOM MATCH & ODDS PREDICTION CALCULATOR (+EV)")
    print("=" * 65)

    while True:
        print("\nEnter Match Details (or type 'q' to quit):\n")

        home_team = input("Enter Home Team Name (e.g., Arsenal): ").strip()
        if home_team.lower() == "q":
            break

        away_team = input("Enter Away Team Name (e.g., Chelsea): ").strip()
        if away_team.lower() == "q":
            break

        try:
            home_odds = float(
                input(f"Enter Bookmaker Odds for {home_team} Win (e.g., 2.10): ")
            )
        except ValueError:
            print("❌ Invalid odds! Please enter a numeric decimal value (e.g., 1.95).")
            continue

        # Extract features and predict
        X_features = get_team_features(home_team, away_team)
        prob = model.predict_proba(X_features)[0]

        prob_home = prob[1]  # Home Win Probability
        implied_prob = 1.0 / home_odds if home_odds > 1.0 else 0.0

        # EV = (Model Prob * Decimal Odds) - 1
        ev = (prob_home * home_odds) - 1.0
        ev_percent = ev * 100.0

        # Display Summary Table
        print("\n" + "-" * 65)
        print(f" MATCH PREDICTION: {home_team.upper()} vs {away_team.upper()}")
        print("-" * 65)
        print(f" Model Predicted Win Prob : {prob_home * 100:.1f}%")
        print(
            f" Bookmaker Implied Prob   : {implied_prob * 100:.1f}% (Odds: {home_odds})"
        )
        print(f" Expected Value (+EV)     : {ev_percent:+.2f}%")

        if ev > 0:
            print(f"\n 🔥 RECOMMENDATION: BET ON {home_team.upper()} (+EV Bet!)")
        else:
            print(f"\n ⛔ RECOMMENDATION: SKIP / NO BET (-EV Bet)")
        print("-" * 65)

        again = input("\nPredict another match? (y/n): ").strip().lower()
        if again != "y":
            print("\nExiting calculator. Good luck!")
            break


if __name__ == "__main__":
    run_user_prediction()
