"""Fetch historical EOD stock prices via yfinance."""
import sys
import os
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TICKERS, START_DATE
from database.db import get_connection, init_db


def fetch_stock_data():
    """Download EOD OHLCV data for all tickers and store in SQLite."""
    init_db()

    ticker_list = list(TICKERS.keys())
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"Downloading EOD data for {len(ticker_list)} tickers: {ticker_list}")
    print(f"Date range: {START_DATE} to {end_date}")

    raw = yf.download(
        ticker_list,
        start=START_DATE,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if raw.empty:
        print("ERROR: No stock data returned from yfinance.")
        return

    rows = []
    for ticker in ticker_list:
        try:
            if len(ticker_list) > 1:
                df = raw[ticker].copy()
            else:
                df = raw.copy()

            df = df.dropna(subset=["Close"])
            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                rows.append({
                    "date": date_str,
                    "ticker": ticker,
                    "open": round(float(row.get("Open", 0)), 4),
                    "high": round(float(row.get("High", 0)), 4),
                    "low": round(float(row.get("Low", 0)), 4),
                    "close": round(float(row.get("Close", 0)), 4),
                    "adj_close": round(float(row.get("Adj Close", row.get("Close", 0))), 4),
                    "volume": int(row.get("Volume", 0)),
                })
            print(f"  {ticker}: {len(df)} trading days")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    if not rows:
        print("ERROR: No rows to save.")
        return

    result_df = pd.DataFrame(rows)

    # Safety check: verify we got reasonable amount of data before replacing
    # This prevents data loss if yfinance has a partial failure
    conn = get_connection()
    try:
        existing = pd.read_sql("SELECT COUNT(*) as cnt FROM stock_prices", conn).iloc[0]["cnt"]
    except Exception:
        existing = 0

    if existing > 0 and len(result_df) < existing * 0.8:
        print(f"SAFETY ABORT: New data ({len(result_df)} rows) is <80% of existing ({existing} rows).")
        print("This likely indicates a yfinance failure. Keeping existing data.")
        conn.close()
        return

    result_df.to_sql("stock_prices", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nSaved {len(result_df)} total rows to stock_prices table")
    print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    print(f"Tickers: {sorted(result_df['ticker'].unique())}")


if __name__ == "__main__":
    fetch_stock_data()
