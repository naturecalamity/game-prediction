import os
from pathlib import Path
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Universal Sports Predictor & Match Evaluator",
    page_icon="🏆",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------------
# SPORT CONFIGURATIONS
# -------------------------------------------------------------
SPORT_CONFIGS = {
    "⚽ Soccer": {
        "csv_file": "raw_soccer_matches.csv",
        "min_odds": 1.50,
        "max_odds": 4.00,
        "win_threshold": 0.50,
        "home_col": "teams.home.name",
        "away_col": "teams.away.name",
        "home_score": "goals.home",
        "away_score": "goals.away",
        "default_home": "Arsenal",
        "default_away": "Chelsea",
    },
    "🏀 Basketball (NBA)": {
        "csv_file": "raw_basketball_matches.csv",
        "min_odds": 1.35,
        "max_odds": 3.00,
        "win_threshold": 0.55,
        "home_col": "teams.home.name",
        "away_col": "teams.away.name",
        "home_score": "scores.home.total",
        "away_score": "scores.away.total",
        "default_home": "L.A. Lakers",
        "default_away": "Boston Celtics",
    },
    "🏈 Football (NFL)": {
        "csv_file": "raw_nfl_matches.csv",
        "min_odds": 1.40,
        "max_odds": 3.50,
        "win_threshold": 0.52,
        "home_col": "teams.home.name",
        "away_col": "teams.away.name",
        "home_score": "scores.home.total",
        "away_score": "scores.away.total",
        "default_home": "Kansas City Chiefs",
        "default_away": "Philadelphia Eagles",
    },
    "⚾ Baseball (MLB)": {
        "csv_file": "raw_baseball_matches.csv",
        "min_odds": 1.45,
        "max_odds": 2.80,
        "win_threshold": 0.53,
        "home_col": "teams.home.name",
        "away_col": "teams.away.name",
        "home_score": "scores.home.total",
        "away_score": "scores.away.total",
        "default_home": "New York Yankees",
        "default_away": "LA Dodgers",
    },
    "🏒 Ice Hockey (NHL)": {
        "csv_file": "raw_nhl_matches.csv",
        "min_odds": 1.50,
        "max_odds": 3.20,
        "win_threshold": 0.52,
        "home_col": "teams.home.name",
        "away_col": "teams.away.name",
        "home_score": "scores.home.total",
        "away_score": "scores.away.total",
        "default_home": "Boston Bruins",
        "default_away": "New York Rangers",
    },
}


# -------------------------------------------------------------
# DATA LOADERS & AUTOMATED STATS ENGINES
# -------------------------------------------------------------
@st.cache_data
def load_sport_dataset(csv_filename):
    csv_path = DATA_DIR / csv_filename
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def compute_automated_h2h_stats(df, home_team, away_team, cfg):
    """Auto-scans dataset for direct past encounters if available."""
    if df is None or df.empty:
        return None

    h_col, a_col = cfg["home_col"], cfg["away_col"]
    hs_col, as_col = cfg["home_score"], cfg["away_score"]

    encounters = df[
        ((df[h_col] == home_team) & (df[a_col] == away_team))
        | ((df[h_col] == away_team) & (df[a_col] == home_team))
    ]

    total_matches = len(encounters)
    if total_matches == 0:
        return None

    home_wins, away_wins, draws = 0, 0, 0
    for _, match in encounters.iterrows():
        h_s, a_s = match.get(hs_col, 0), match.get(as_col, 0)
        if match[h_col] == home_team:
            if h_s > a_s:
                home_wins += 1
            elif a_s > h_s:
                away_wins += 1
            else:
                draws += 1
        else:
            if a_s > h_s:
                home_wins += 1
            elif h_s > a_s:
                away_wins += 1
            else:
                draws += 1

    return {
        "total": total_matches,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "draws": draws,
    }


def compute_prediction_metrics(
    home_odds,
    away_odds,
    h2h_home_wins,
    h2h_away_wins,
    total_encounters,
    home_current_rank,
    away_current_rank,
    home_past_rank,
    away_past_rank,
):
    """Calculates composite win probabilities factoring in odds, H2H, and table momentum."""
    # 1. Market Implied Fair Probabilities (Removing vigorish)
    raw_prob_home = 1.0 / home_odds
    raw_prob_away = 1.0 / away_odds
    total_implied = raw_prob_home + raw_prob_away
    fair_prob_home = raw_prob_home / total_implied

    # 2. H2H Rate Factor
    if total_encounters > 0:
        h2h_rate_home = h2h_home_wins / total_encounters
    else:
        h2h_rate_home = 0.50

    # 3. Table Standings & Momentum Factor
    # Current Rank Edge (positive if Home is ranked higher/lower number)
    current_rank_diff = away_current_rank - home_current_rank
    current_table_factor = max(-0.10, min(0.10, current_rank_diff * 0.02))

    # Momentum/Trajectory Edge (Did team improve compared to past rank?)
    home_trajectory = home_past_rank - home_current_rank  # Positive = improving
    away_trajectory = away_past_rank - away_current_rank  # Positive = improving
    momentum_diff = home_trajectory - away_trajectory
    momentum_factor = max(-0.05, min(0.05, momentum_diff * 0.01))

    total_table_factor = current_table_factor + momentum_factor

    # Composite Probability: 55% Odds + 25% Direct Encounters + 20% Standings & Momentum
    prob_home = (
        (fair_prob_home * 0.55)
        + (h2h_rate_home * 0.25)
        + ((0.50 + total_table_factor) * 0.20)
    )
    prob_home = max(0.01, min(0.99, prob_home))
    prob_away = 1.0 - prob_home

    # Expected Value (+EV)
    ev_home_pct = ((prob_home * home_odds) - 1.0) * 100.0
    ev_away_pct = ((prob_away * away_odds) - 1.0) * 100.0

    return prob_home, prob_away, ev_home_pct, ev_away_pct, h2h_rate_home


# -------------------------------------------------------------
# DASHBOARD UI
# -------------------------------------------------------------
st.title("🏆 Match Outcome & Standings-Based Predictor")
st.markdown(
    "Input target games, past match encounter results, bookmaker odds, and past/current standings to compute winning predictions."
)

selected_sport = st.selectbox(
    "Choose Sport Category", list(SPORT_CONFIGS.keys()), index=0
)
cfg = SPORT_CONFIGS[selected_sport]

df_hist = load_sport_dataset(cfg["csv_file"])

st.divider()

# TAB NAVIGATION
tab_input, tab_batch = st.tabs(
    ["🎯 Custom Fixture Evaluator", "📋 Bulk Dataset Scanner"]
)

# -------------------------------------------------------------
# TAB 1: CUSTOM FIXTURE EVALUATOR (User Inputs)
# -------------------------------------------------------------
with tab_input:
    st.subheader("1️⃣ Teams & Bookmaker Odds")
    c1, c2 = st.columns(2)

    with c1:
        home_team = st.text_input("Home Team / Player 1", value=cfg["default_home"])
        home_odds = st.number_input(
            f"Odds for {home_team}", min_value=1.01, value=2.10, step=0.05
        )

    with c2:
        away_team = st.text_input("Away Team / Player 2", value=cfg["default_away"])
        away_odds = st.number_input(
            f"Odds for {away_team}", min_value=1.01, value=1.85, step=0.05
        )

    # Check for auto H2H
    auto_h2h = compute_automated_h2h_stats(df_hist, home_team, away_team, cfg)

    st.subheader("2️⃣ Past Match Encounter Statistics (Head-to-Head)")
    if auto_h2h:
        st.success(
            f"🔍 **Auto-Detected History from Dataset:** {auto_h2h['total']} past encounters found "
            f"({auto_h2h['home_wins']} wins for {home_team}, {auto_h2h['away_wins']} wins for {away_team}, {auto_h2h['draws']} draws)."
        )

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        default_encounters = auto_h2h["total"] if auto_h2h else 6
        total_encounters = st.number_input(
            "Total Past Encounters Played",
            min_value=0,
            value=default_encounters,
            step=1,
        )

    with h_col2:
        default_home_w = auto_h2h["home_wins"] if auto_h2h else min(4, total_encounters)
        h2h_home_wins = st.number_input(
            f"Past Wins by {home_team}",
            min_value=0,
            max_value=int(total_encounters),
            value=min(int(default_home_w), int(total_encounters)),
            step=1,
        )

    with h_col3:
        max_away_possible = total_encounters - h2h_home_wins
        default_away_w = (
            auto_h2h["away_wins"] if auto_h2h else min(2, max_away_possible)
        )
        h2h_away_wins = st.number_input(
            f"Past Wins by {away_team}",
            min_value=0,
            max_value=int(max_away_possible),
            value=min(int(default_away_w), int(max_away_possible)),
            step=1,
        )

    st.subheader("3️⃣ League Standings Position (Current vs Past Season/Period)")
    st.caption(
        "Lower rank number means higher standing (e.g., Rank #1 is Top of the Table)."
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(f"**🏠 {home_team} Table Standings**")
        home_current_rank = st.number_input(
            f"{home_team} Current Rank", min_value=1, value=2, step=1
        )
        home_past_rank = st.number_input(
            f"{home_team} Past Season/Period Rank", min_value=1, value=4, step=1
        )

    with t_col2:
        st.markdown(f"**✈️ {away_team} Table Standings**")
        away_current_rank = st.number_input(
            f"{away_team} Current Rank", min_value=1, value=6, step=1
        )
        away_past_rank = st.number_input(
            f"{away_team} Past Season/Period Rank", min_value=1, value=5, step=1
        )

    st.divider()
    analyze_btn = st.button(
        "🚀 Predict Outcome & Likely Winner", use_container_width=True
    )

    if analyze_btn:
        prob_home, prob_away, ev_home, ev_away, h2h_rate = compute_prediction_metrics(
            home_odds,
            away_odds,
            h2h_home_wins,
            h2h_away_wins,
            total_encounters,
            home_current_rank,
            away_current_rank,
            home_past_rank,
            away_past_rank,
        )

        st.subheader(f"Match Analysis: {home_team.upper()} vs {away_team.upper()}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{home_team} Win Prob", f"{prob_home * 100:.1f}%")
        m2.metric(f"{away_team} Win Prob", f"{prob_away * 100:.1f}%")
        m3.metric(f"{home_team} H2H Rate", f"{h2h_rate * 100:.1f}%")
        m4.metric(f"{home_team} Expected Value", f"{ev_home:+.2f}%")

        st.divider()

        # Decision Engine Rules
        min_o, max_o, prob_t = cfg["min_odds"], cfg["max_odds"], cfg["win_threshold"]
        has_table_edge = home_current_rank <= away_current_rank
        has_h2h_edge = (
            (h2h_home_wins >= h2h_away_wins) if total_encounters > 0 else True
        )

        if home_odds > max_o or home_odds < min_o:
            st.warning(
                f"⚠️ **RECOMMENDATION: SKIP BET ON {home_team.upper()}**\n\n"
                f"Odds of **{home_odds:.2f}** fall outside the defined safe range (**{min_o:.2f} - {max_o:.2f}**)."
            )
        elif prob_home >= prob_t and has_table_edge and has_h2h_edge:
            st.success(
                f"🔥 **RECOMMENDATION: LIKELY WINNER — {home_team.upper()}**\n\n"
                f"1. **Adjusted Victory Probability:** {prob_home * 100:.1f}% (Threshold: ≥ {prob_t * 100:.0f}%)\n"
                f"2. **Table Standings Advantage:** Rank **#{home_current_rank}** vs **#{away_current_rank}** (Past Rank: #{home_past_rank} vs #{away_past_rank})\n"
                f"3. **Head-to-Head Encounters:** Won **{h2h_home_wins} of {total_encounters}** matches ({h2h_rate * 100:.0f}% rate)\n"
                f"4. **Bookmaker Value (+EV):** {ev_home:+.2f}%\n\n"
                f"👉 **Action:** Bet on **{home_team}**."
            )
        elif prob_away >= prob_t and away_current_rank < home_current_rank:
            st.success(
                f"🔥 **RECOMMENDATION: LIKELY WINNER — {away_team.upper()}**\n\n"
                f"1. **Adjusted Victory Probability:** {prob_away * 100:.1f}%\n"
                f"2. **Table Standings Advantage:** {away_team} holds superior rank (**#{away_current_rank}** vs **#{home_current_rank}**)\n"
                f"3. **Expected Value (+EV):** {ev_away:+.2f}%\n\n"
                f"👉 **Action:** Bet on **{away_team}**."
            )
        else:
            st.error(
                f"⛔ **RECOMMENDATION: SKIP BET**\n\n"
                f"Neither team fully satisfies the required winning criteria (Table edge, H2H dominance, and minimum win probability)."
            )

# -------------------------------------------------------------
# TAB 2: BULK DATASET SCANNER
# -------------------------------------------------------------
with tab_batch:
    st.subheader("📋 Dataset Fixture Scanner")
    if df_hist is None or df_hist.empty:
        st.info(
            "No CSV dataset loaded. Use Tab 1 for custom manual fixture evaluations."
        )
    else:
        st.write(
            f"Loaded **{len(df_hist)}** fixtures from history for {selected_sport}."
        )
        st.dataframe(df_hist.head(20), use_container_width=True)
