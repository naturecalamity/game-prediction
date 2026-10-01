import datetime
import numpy as np
import pandas as pd


def compute_lookback_features(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    target_date: str,
    mode: str = "last_5_games",
):
    """Calculates features based on the user's selected lookback mode:

    'last_5_games' or 'last_5_years'
    """
    df["match_date"] = pd.to_datetime(df["match_date"])
    target_dt = pd.to_datetime(target_date)

    # Filter data strictly BEFORE target_date (prevents data leakage)
    past_df = df[df["match_date"] < target_dt].copy()

    if mode == "last_5_years":
        # Filter games in the last 5 years (1825 days)
        start_date = target_dt - datetime.timedelta(days=1825)
        past_df = past_df[past_df["match_date"] >= start_date]

        # Extract matches involving home/away team
        home_matches = past_df[
            (past_df["home_team"] == home_team) | (past_df["away_team"] == home_team)
        ]
        away_matches = past_df[
            (past_df["home_team"] == away_team) | (past_df["away_team"] == away_team)
        ]

    else:  # Default: 'last_5_games'
        # Filter exact last 5 matches for each team
        home_matches = past_df[
            (past_df["home_team"] == home_team) | (past_df["away_team"] == home_team)
        ].tail(5)
        away_matches = past_df[
            (past_df["home_team"] == away_team) | (past_df["away_team"] == away_team)
        ].tail(5)

    # 1. Compute Win Rate & Points Per Game (PPG) over the window
    def calc_ppg(team, matches):
        if matches.empty:
            return 1.3  # Baseline default
        points = 0
        for _, row in matches.iterrows():
            is_home = row["home_team"] == team
            res = row["result"]
            if (is_home and res == "H") or (not is_home and res == "A"):
                points += 3
            elif res == "D":
                points += 1
        return points / len(matches)

    home_ppg = calc_ppg(home_team, home_matches)
    away_ppg = calc_ppg(away_team, away_matches)

    # 2. Extract latest Elo from the filtered window
    home_elo = home_matches["home_elo"].iloc[-1] if not home_matches.empty else 1500.0
    away_elo = away_matches["away_elo"].iloc[-1] if not away_matches.empty else 1500.0

    return {
        "home_ppg": round(home_ppg, 2),
        "away_ppg": round(away_ppg, 2),
        "elo_diff": round(home_elo - away_elo, 2),
        "sample_size_home": len(home_matches),
        "sample_size_away": len(away_matches),
    }
