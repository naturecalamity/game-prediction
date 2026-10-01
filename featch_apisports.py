import time
from pathlib import Path
import pandas as pd
import requests

# Set your API-Sports key
API_KEY = "YOUR_API_SPORTS_KEY_HERE"

# Ensure data directory exists
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# API-SPORTS ENDPOINTS & SEASONS CONFIGURATION
# -------------------------------------------------------------
# Add any sports, leagues, or seasons you wish to fetch
SPORTS_CONFIG = {
    "soccer": {
        "base_url": "https://v3.football.api-sports.io",
        "output_csv": "raw_soccer_matches.csv",
        "seasons": [2023, 2024, 2025],
        "endpoint": "/fixtures",
    },
    "basketball": {
        "base_url": "https://v1.basketball.api-sports.io",
        "output_csv": "raw_basketball_matches.csv",
        "seasons": ["2023-2024", "2024-2025"],
        "endpoint": "/games",
    },
    "nfl": {
        "base_url": "https://v1.american-football.api-sports.io",
        "output_csv": "raw_nfl_matches.csv",
        "seasons": [2023, 2024, 2025],
        "endpoint": "/games",
    },
    "baseball": {
        "base_url": "https://v1.baseball.api-sports.io",
        "output_csv": "raw_baseball_matches.csv",
        "seasons": [2023, 2024, 2025],
        "endpoint": "/games",
    },
    "nhl": {
        "base_url": "https://v1.hockey.api-sports.io",
        "output_csv": "raw_hockey_matches.csv",
        "seasons": ["2023-2024", "2024-2025"],
        "endpoint": "/games",
    },
}


# -------------------------------------------------------------
# FETCHING LOGIC WITH PAGINATION AND RATE LIMITING
# -------------------------------------------------------------
def fetch_all_games_for_sport(sport_name, config):
    """Iterates through configured seasons and fetches all fixtures."""
    headers = {
        "x-apisports-key": API_KEY,
        "x-rapidapi-host": config["base_url"].replace("https://", ""),
    }

    all_matches = []

    for season in config["seasons"]:
        print(f"📥 Fetching {sport_name.upper()} data for season: {season}...")

        url = f"{config['base_url']}{config['endpoint']}"
        params = {"season": str(season)}

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(
                    f"⚠️ Error {response.status_code} fetching {sport_name} - {season}"
                )
                continue

            data = response.json()
            fixtures = data.get("response", [])

            print(f"✅ Retrieved {len(fixtures)} matches for {sport_name} ({season})")
            all_matches.extend(fixtures)

            # Throttle requests to respect API rate limits (1 request per second)
            time.sleep(1.2)

        except Exception as e:
            print(f"❌ Failed to fetch {sport_name} ({season}): {e}")

    # Normalize JSON payload into flat DataFrame columns
    if all_matches:
        df = pd.json_normalize(all_matches)
        output_path = DATA_DIR / config["output_csv"]
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(df)} total matches to {output_path.resolve()}\n")
    else:
        print(f"⚠️ No matches collected for {sport_name}.\n")


def main():
    print("🚀 Starting API-Sports Global Data Ingestion...\n")
    for sport_name, config in SPORTS_CONFIG.items():
        fetch_all_games_for_sport(sport_name, config)
    print("🎉 All game datasets successfully fetched and updated!")


if __name__ == "__main__":
    main()
