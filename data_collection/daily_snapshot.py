"""Daily EOD data collection script.

Run once per day (e.g., via cron at 11:59 PM UTC or GitHub Actions).
Fetches latest EOD stock prices, Google Trends, and contest data,
then recomputes the LAI.
"""
import sys
import os
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_db, get_connection
from data_collection.stock_data import fetch_stock_data
from data_collection.google_trends import fetch_and_store
from data_collection.contest_scraper import scrape_all_contests
from indicator.lai_calculator import compute_lai


def daily_update():
    """Run the full daily EOD pipeline."""
    init_db()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"=== Daily LAI Update: {today} ===\n")

    # Step 1: Update stock prices (EOD)
    print("[1/4] Updating stock prices...")
    try:
        fetch_stock_data()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 2: Update Google Trends
    print("\n[2/4] Updating Google Trends...")
    try:
        fetch_and_store()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 3: Update contest data (if any new contests)
    print("\n[3/4] Updating contest data...")
    try:
        scrape_all_contests()
    except Exception as e:
        print(f"  ERROR: {e}")

    # Step 4: Recompute LAI
    print("\n[4/4] Recomputing LAI...")
    try:
        result = compute_lai()
        latest = result.dropna(subset=["lai_smoothed"]).iloc[-1]
        print(f"\n{'='*50}")
        print(f"TODAY'S LAI: {latest['lai_smoothed']:.1f}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n=== Daily update complete: {today} ===")


if __name__ == "__main__":
    daily_update()
