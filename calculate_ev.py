import os
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_soccer.json"
HISTORICAL_CSV = BASE_DIR / "data" / "raw_soccer_matches.csv"


def load_soccer_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first!"
        )
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def get_team_features(home_team, away_team):
    """Calculates rolling stats from historical matches CSV."""
    if not os.path.exists(HISTORICAL_CSV):
        raise FileNotFoundError(
            f"Historical CSV not found at {HISTORICAL_CSV}. Run featch_apisports.py first!"
        )

    df = pd.read_csv(HISTORICAL_CSV)
    df["home_score"] = df.get("goals.home", 0)
    df["away_score"] = df.get("goals.away", 0)

    # Defaults
    global_st_home = df["home_score"].tail(5).mean()
    global_st_away = df["away_score"].tail(5).mean()
    global_lt_home = df["home_score"].tail(30).mean()
    global_lt_away = df["away_score"].tail(30).mean()

    # Home stats
    home_matches = df[
        df["teams.home.name"].str.contains(home_team, case=False, na=False)
    ]
    st_home = (
        home_matches["home_score"].tail(5).mean()
        if not home_matches.empty
        else global_st_home
    )
    lt_home = (
        home_matches["home_score"].tail(30).mean()
        if not home_matches.empty
        else global_lt_home
    )

    # Away stats
    away_matches = df[
        df["teams.away.name"].str.contains(away_team, case=False, na=False)
    ]
    st_away = (
        away_matches["away_score"].tail(5).mean()
        if not away_matches.empty
        else global_st_away
    )
    lt_away = (
        away_matches["away_score"].tail(30).mean()
        if not away_matches.empty
        else global_lt_away
    )

    return pd.DataFrame(
        [
            {
                "st_home_goals": st_home,
                "st_away_goals": st_away,
                "lt_home_goals": lt_home,
                "lt_away_goals": lt_away,
                "home_form_ratio": st_home / (lt_home + 1e-5),
                "away_form_ratio": st_away / (lt_away + 1e-5),
            }
        ]
    )


def calculate_ev_bets():
    model = load_soccer_model()

    print("=" * 65)
    print("        EXPECTED VALUE (+EV) CALCULATOR")
    print("=" * 65)

    home_team = input("Enter Home Team: ").strip()
    away_team = input("Enter Away Team: ").strip()

    try:
        home_odds = float(input(f"Enter Bookmaker Odds for {home_team} Win: "))
    except ValueError:
        print("Invalid odds input. Must be a decimal number (e.g. 1.85).")
        return

    # Engineer features and predict
    features = get_team_features(home_team, away_team)
    prob_home = model.predict_proba(features)[0][1]
    implied_prob = 1.0 / home_odds if home_odds > 1.0 else 0.0

    # EV Formula: (Probability * Odds) - 1
    ev = (prob_home * home_odds) - 1.0
    ev_percent = ev * 100.0

    print("\n" + "-" * 65)
    print(f" MATCH: {home_team} vs {away_team}")
    print("-" * 65)
    print(f" Model Probability : {prob_home * 100:.1f}%")
    print(f" Implied Odds Prob : {implied_prob * 100:.1f}% (Odds: {home_odds})")
    print(f" Expected Value    : {ev_percent:+.2f}%")

    if ev > 0:
        print(f"\n 🔥 RECOMMENDATION: +EV BET ON {home_team.upper()}")
    else:
        print(f"\n ⛔ RECOMMENDATION: SKIP (-EV BET)")
    print("-" * 65)


if __name__ == "__main__":
    calculate_ev_bets()
