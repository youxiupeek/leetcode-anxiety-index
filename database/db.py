"""SQLite database access layer for LAI."""
import sqlite3
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS contest_participation (
            contest_slug TEXT PRIMARY KEY,
            contest_type TEXT,
            contest_number INTEGER,
            contest_date TEXT,
            participant_count INTEGER
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS google_trends (
            date TEXT PRIMARY KEY,
            leetcode_interest REAL,
            tech_layoffs_interest REAL,
            coding_interview_interest REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT,
            ticker TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            PRIMARY KEY (date, ticker)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lai_values (
            date TEXT PRIMARY KEY,
            lai_raw REAL,
            lai_smoothed REAL,
            contest_component REAL,
            trends_component REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS leetcode_snapshots (
            date TEXT PRIMARY KEY,
            total_submissions_cumulative INTEGER,
            daily_delta INTEGER,
            problem_count INTEGER
        )
    """)

    conn.commit()
    conn.close()


def save_dataframe(df, table_name, if_exists="replace"):
    conn = get_connection()
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()


def load_dataframe(table_name, parse_dates=None):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)
    conn.close()
    return df


def upsert_contest(slug, ctype, number, date_str, count):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO contest_participation
           (contest_slug, contest_type, contest_number, contest_date, participant_count)
           VALUES (?, ?, ?, ?, ?)""",
        (slug, ctype, number, date_str, count),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
