"""Fetch Google Trends daily data for 'leetcode' and related terms."""
import sys
import os
import time
import pandas as pd
import numpy as np
from pytrends.request import TrendReq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import START_DATE
from database.db import get_connection, init_db


def fetch_weekly_trends(keywords, geo="US", start="2020-01-01"):
    """Fetch weekly Google Trends data using pytrends."""
    pytrends = TrendReq(hl="en-US", tz=360)
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    timeframe = f"{start} {end}"

    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df


def interpolate_weekly_to_daily(weekly_df):
    """Linearly interpolate weekly data to daily resolution."""
    daily_idx = pd.date_range(start=weekly_df.index.min(), end=weekly_df.index.max(), freq="D")
    daily_df = weekly_df.reindex(daily_idx)
    daily_df = daily_df.interpolate(method="linear")
    daily_df.index.name = "date"
    return daily_df


def fetch_and_store():
    """Main entry: fetch Google Trends data and store in SQLite."""
    init_db()

    print("Fetching Google Trends weekly data...")
    keywords = ["leetcode", "tech layoffs", "coding interview"]

    try:
        weekly = fetch_weekly_trends(keywords, geo="US", start=START_DATE)
    except Exception as e:
        print(f"Error fetching trends: {e}")
        print("Retrying with smaller time windows...")
        weekly = _fetch_in_chunks(keywords, START_DATE)

    if weekly.empty:
        print("ERROR: No Google Trends data returned.")
        return

    print(f"Got {len(weekly)} weekly data points from {weekly.index.min()} to {weekly.index.max()}")

    # Interpolate to daily
    daily = interpolate_weekly_to_daily(weekly)
    print(f"Interpolated to {len(daily)} daily data points")

    # Rename columns for DB
    col_map = {
        "leetcode": "leetcode_interest",
        "tech layoffs": "tech_layoffs_interest",
        "coding interview": "coding_interview_interest",
    }
    daily = daily.rename(columns=col_map)

    # Save to database
    daily = daily.reset_index()
    daily = daily.rename(columns={"index": "date"})
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    # Safety check: don't replace if new data is significantly smaller
    conn = get_connection()
    try:
        existing = pd.read_sql("SELECT COUNT(*) as cnt FROM google_trends", conn).iloc[0]["cnt"]
    except Exception:
        existing = 0

    if existing > 0 and len(daily) < existing * 0.8:
        print(f"SAFETY ABORT: New data ({len(daily)} rows) is <80% of existing ({existing} rows).")
        print("This likely indicates a pytrends failure. Keeping existing data.")
        conn.close()
        return

    daily.to_sql("google_trends", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(daily)} rows to google_trends table")
    print(f"Sample:\n{daily.head()}")
    print(f"\nDate range: {daily['date'].min()} to {daily['date'].max()}")


def _fetch_in_chunks(keywords, start_date):
    """Fetch trends in overlapping 12-month chunks and rescale."""
    frames = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp.now()

    pytrends = TrendReq(hl="en-US", tz=360)

    while start < end:
        chunk_end = min(start + pd.DateOffset(months=12), end)
        timeframe = f"{start.strftime('%Y-%m-%d')} {chunk_end.strftime('%Y-%m-%d')}"
        print(f"  Fetching chunk: {timeframe}")

        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="US")
            chunk = pytrends.interest_over_time()
            if "isPartial" in chunk.columns:
                chunk = chunk.drop(columns=["isPartial"])
            frames.append(chunk)
        except Exception as e:
            print(f"  Error on chunk {timeframe}: {e}")

        start = chunk_end - pd.DateOffset(weeks=2)  # overlap for rescaling
        time.sleep(5)  # respect rate limits

    if not frames:
        return pd.DataFrame()

    # Simple concatenation (overlapping periods averaged)
    combined = pd.concat(frames)
    combined = combined.groupby(combined.index).mean()
    combined = combined.sort_index()

    return combined


if __name__ == "__main__":
    fetch_and_store()
