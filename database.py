import sqlite3
import pandas as pd
from rapidfuzz import process, fuzz

DB_FILE = "sports_ai.db"

def init_db():
    """Initializes the database tables for teams, historical odds, and features."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Canonical teams table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS team_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_name TEXT UNIQUE,
        alias TEXT UNIQUE
    )
    """)
    
    # 2. Master odds table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_odds (
        match_id TEXT PRIMARY KEY,
        sport TEXT,
        match_date TEXT,
        home_team TEXT,
        away_team TEXT,
        home_odds REAL,
        draw_odds REAL,
        away_odds REAL,
        fair_prob_home REAL,
        fair_prob_away REAL,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

def resolve_team(raw_name: str, score_threshold: float = 85.0) -> str:
    """Maps any incoming raw team name to a canonical name using DB lookups and Fuzzy Matching."""
    raw_clean = raw_name.strip()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check 1: Direct alias match in DB
    cursor.execute("SELECT canonical_name FROM team_aliases WHERE alias = ?", (raw_clean,))
    row = cursor.fetchone()
    if row:
        conn.close()
        return row[0]

    # Check 2: Fuzzy match against existing canonical names
    cursor.execute("SELECT DISTINCT canonical_name FROM team_aliases")
    known_teams = [r[0] for r in cursor.fetchall()]

    if known_teams:
        match, score, _ = process.extractOne(raw_clean, known_teams, scorer=fuzz.token_sort_ratio)
        if score >= score_threshold:
            # Save alias for future O(1) performance
            cursor.execute("INSERT OR IGNORE INTO team_aliases (canonical_name, alias) VALUES (?, ?)", (match, raw_clean))
            conn.commit()
            conn.close()
            return match

    # Check 3: If brand new, save as its own canonical name
    cursor.execute("INSERT OR IGNORE INTO team_aliases (canonical_name, alias) VALUES (?, ?)", (raw_clean, raw_clean))
    conn.commit()
    conn.close()
    return raw_clean

init_db()