"""Core LAI (LeetCode Anxiety Index) computation engine."""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    WEIGHT_TRENDS, WEIGHT_CONTEST, ROLLING_WINDOW, EMA_SPAN,
    THRESHOLD_LOW, THRESHOLD_HIGH, BENCHMARK,
)
from database.db import get_connection, init_db


def rolling_zscore_to_100(series, window=90):
    """Normalize a series using rolling z-score mapped to 0-100.

    Z-score is clipped to [-3, 3] then linearly mapped to [0, 100].
    """
    rolling_mean = series.rolling(window=window, min_periods=max(window // 3, 7)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(window // 3, 7)).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (series - rolling_mean) / rolling_std
    z = z.clip(-3, 3)
    normalized = (z + 3) / 6 * 100
    return normalized


def load_google_trends():
    """Load and prepare Google Trends data."""
    conn = get_connection()
    df = pd.read_sql("SELECT date, leetcode_interest FROM google_trends", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["leetcode_interest"]


def load_contest_data():
    """Load contest participation, interpolate to daily."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT contest_date, participant_count FROM contest_participation ORDER BY contest_date",
        conn,
    )
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["contest_date"] = pd.to_datetime(df["contest_date"])
    df = df.set_index("contest_date").sort_index()

    # Interpolate to daily
    daily_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    daily = df["participant_count"].reindex(daily_idx).interpolate(method="linear")
    daily.index.name = "date"
    return daily


def load_stock_prices():
    """Load stock prices as a pivot table (date x ticker)."""
    conn = get_connection()
    df = pd.read_sql("SELECT date, ticker, adj_close FROM stock_prices", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot_table(index="date", columns="ticker", values="adj_close")
    pivot = pivot.sort_index()
    return pivot


def compute_lai():
    """Compute the full LAI time series and save to database."""
    init_db()

    print("Loading data sources...")
    trends = load_google_trends()
    contests = load_contest_data()
    stocks = load_stock_prices()

    print(f"  Google Trends: {len(trends)} daily points ({trends.index.min()} to {trends.index.max()})")
    if not contests.empty:
        print(f"  Contests: {len(contests)} daily points ({contests.index.min()} to {contests.index.max()})")
    else:
        print("  Contests: NO DATA (scraper may still be running)")
    print(f"  Stocks: {len(stocks)} trading days")

    # --- Build common date index ---
    # Use trading days from stock data as the master index
    trading_days = stocks.index

    # Normalize Google Trends
    trends_norm = rolling_zscore_to_100(trends, window=ROLLING_WINDOW)
    # Align to trading days (forward-fill weekends)
    trends_aligned = trends_norm.reindex(trading_days, method="ffill")

    if not contests.empty:
        # Normalize contest participation
        contests_norm = rolling_zscore_to_100(contests, window=ROLLING_WINDOW)
        contests_aligned = contests_norm.reindex(trading_days, method="ffill")
    else:
        # If no contest data, use trends only
        contests_aligned = pd.Series(50.0, index=trading_days)

    # --- Composite LAI ---
    has_contest = not contests.empty
    if has_contest:
        lai_raw = (WEIGHT_TRENDS * trends_aligned + WEIGHT_CONTEST * contests_aligned)
    else:
        # Trends only until contest data is ready
        lai_raw = trends_aligned

    # Clip to 0-100
    lai_raw = lai_raw.clip(0, 100)

    # EMA smoothing
    lai_smoothed = lai_raw.ewm(span=EMA_SPAN).mean()

    # --- Save to database ---
    result = pd.DataFrame({
        "date": trading_days.strftime("%Y-%m-%d"),
        "lai_raw": lai_raw.values,
        "lai_smoothed": lai_smoothed.values,
        "contest_component": contests_aligned.values if has_contest else [None] * len(trading_days),
        "trends_component": trends_aligned.values,
    })

    conn = get_connection()
    result.to_sql("lai_values", conn, if_exists="replace", index=False)
    conn.close()

    # --- Print summary ---
    valid = result.dropna(subset=["lai_smoothed"])
    print(f"\nLAI computed: {len(valid)} trading days")
    print(f"Date range: {valid['date'].min()} to {valid['date'].max()}")
    latest = valid.iloc[-1]
    print(f"\nLatest LAI: {latest['lai_smoothed']:.1f} (raw: {latest['lai_raw']:.1f})")
    print(f"  Trends component: {latest['trends_component']:.1f}")
    if has_contest:
        print(f"  Contest component: {latest['contest_component']:.1f}")

    # Signal
    val = latest["lai_smoothed"]
    if val >= THRESHOLD_HIGH:
        signal = "ELEVATED ANXIETY"
    elif val <= THRESHOLD_LOW:
        signal = "LOW ANXIETY"
    else:
        signal = "NORMAL"
    print(f"  Signal: {signal}")

    return result


if __name__ == "__main__":
    compute_lai()
