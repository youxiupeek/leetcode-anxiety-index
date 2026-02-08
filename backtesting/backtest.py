"""Backtesting engine for the LeetCode Anxiety Index."""
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BENCHMARK, THRESHOLD_HIGH, THRESHOLD_LOW, LAYOFF_EVENTS, TICKERS
from database.db import get_connection
from indicator.lai_calculator import load_stock_prices


def load_lai():
    """Load LAI time series."""
    conn = get_connection()
    df = pd.read_sql("SELECT date, lai_smoothed, lai_raw, trends_component, contest_component FROM lai_values", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def compute_returns(prices, periods=[1, 5, 10, 20, 60]):
    """Compute forward returns for various holding periods."""
    result = {}
    for p in periods:
        result[f"fwd_{p}d"] = prices.pct_change(p).shift(-p)
    return pd.DataFrame(result, index=prices.index)


def lead_lag_correlation(lai, stock_returns, max_lag=60):
    """Compute correlation between LAI(t) and stock_return(t+lag).

    Positive lag = LAI leads stocks (predictive).
    """
    results = {}
    lai_clean = lai.dropna()
    ret_clean = stock_returns.dropna()
    common = lai_clean.index.intersection(ret_clean.index)
    lai_c = lai_clean.loc[common]
    ret_c = ret_clean.loc[common]

    for lag in range(-max_lag, max_lag + 1):
        shifted = ret_c.shift(-lag)
        valid = ~(lai_c.isna() | shifted.isna())
        if valid.sum() > 30:
            results[lag] = lai_c[valid].corr(shifted[valid])
        else:
            results[lag] = np.nan
    return pd.Series(results)


def granger_causality_test(lai, stock_returns, max_lag=20):
    """Test if LAI Granger-causes stock returns."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests

        data = pd.DataFrame({
            "returns": stock_returns,
            "lai_diff": lai.diff(),
        }).dropna()

        if len(data) < max_lag * 3:
            return {"error": "insufficient data"}

        results = grangercausalitytests(data[["returns", "lai_diff"]], maxlag=max_lag, verbose=False)

        p_values = {}
        for lag_num, res in results.items():
            p_values[lag_num] = round(res[0]["ssr_ftest"][1], 6)
        return p_values
    except Exception as e:
        return {"error": str(e)}


def regime_analysis(lai, stock_returns):
    """Compare stock returns in high-anxiety vs low-anxiety regimes."""
    combined = pd.DataFrame({"lai": lai, "ret": stock_returns}).dropna()

    high = combined[combined["lai"] > THRESHOLD_HIGH]["ret"]
    low = combined[combined["lai"] < THRESHOLD_LOW]["ret"]
    normal = combined[(combined["lai"] >= THRESHOLD_LOW) & (combined["lai"] <= THRESHOLD_HIGH)]["ret"]

    result = {
        "high_anxiety_days": len(high),
        "low_anxiety_days": len(low),
        "normal_days": len(normal),
        "high_anxiety_mean_daily_return_bps": round(high.mean() * 10000, 2) if len(high) > 0 else None,
        "low_anxiety_mean_daily_return_bps": round(low.mean() * 10000, 2) if len(low) > 0 else None,
        "normal_mean_daily_return_bps": round(normal.mean() * 10000, 2) if len(normal) > 0 else None,
        "high_anxiety_annualized_return_pct": round(high.mean() * 252 * 100, 2) if len(high) > 0 else None,
        "low_anxiety_annualized_return_pct": round(low.mean() * 252 * 100, 2) if len(low) > 0 else None,
    }

    if len(high) > 5 and len(low) > 5:
        t_stat, p_value = stats.ttest_ind(high, low, equal_var=False)
        result["t_statistic"] = round(t_stat, 4)
        result["p_value"] = round(p_value, 6)

    return result


def forward_return_analysis(lai, stock_prices, threshold=THRESHOLD_HIGH, periods=[5, 10, 20, 60]):
    """After LAI crosses above threshold, compute forward returns."""
    # Find dates where LAI crosses above threshold
    above = lai > threshold
    crossings = above & ~above.shift(1, fill_value=False)
    signal_dates = crossings[crossings].index

    results = {}
    for period in periods:
        fwd_returns = []
        for date in signal_dates:
            future = date + pd.Timedelta(days=period)
            # Find closest trading day
            mask = stock_prices.index >= future
            if mask.any():
                future_date = stock_prices.index[mask][0]
                mask_now = stock_prices.index >= date
                if mask_now.any():
                    now_date = stock_prices.index[mask_now][0]
                    ret = stock_prices.loc[future_date] / stock_prices.loc[now_date] - 1
                    fwd_returns.append(float(ret))

        if fwd_returns:
            results[f"{period}d"] = {
                "n_signals": len(fwd_returns),
                "mean_return_pct": round(np.mean(fwd_returns) * 100, 2),
                "median_return_pct": round(np.median(fwd_returns) * 100, 2),
                "win_rate_pct": round(np.mean([r > 0 for r in fwd_returns]) * 100, 1),
                "std_pct": round(np.std(fwd_returns) * 100, 2),
                "sharpe": round(np.mean(fwd_returns) / np.std(fwd_returns), 2) if np.std(fwd_returns) > 0 else 0,
            }

    return results


def event_study(lai, stock_prices, events=None, window=60):
    """Event study around known layoff dates."""
    if events is None:
        events = LAYOFF_EVENTS

    results = {}
    for date_str, label in events.items():
        event_date = pd.Timestamp(date_str)
        # Find closest trading day
        mask = stock_prices.index >= event_date
        if not mask.any():
            continue
        td = stock_prices.index[mask][0]

        # LAI around event
        lai_window = lai.loc[td - pd.Timedelta(days=30): td + pd.Timedelta(days=window)]
        if lai_window.empty:
            continue

        # Stock return in [0, +window] days
        future_mask = stock_prices.index >= td
        future_prices = stock_prices.loc[future_mask].head(window + 1)
        if len(future_prices) < 2:
            continue

        cum_return = future_prices.iloc[-1] / future_prices.iloc[0] - 1

        results[label] = {
            "date": date_str,
            "lai_at_event": round(float(lai.loc[td]), 1) if td in lai.index else None,
            "lai_30d_before": round(float(lai.loc[lai.loc[:td].index[-30]]), 1) if len(lai.loc[:td]) >= 30 else None,
            f"stock_{window}d_return_pct": round(float(cum_return) * 100, 2),
        }

    return results


def rolling_correlation(lai, stock_returns, windows=[30, 60, 90]):
    """Compute rolling correlation between LAI and stock returns."""
    combined = pd.DataFrame({"lai": lai, "ret": stock_returns}).dropna()
    result = {}
    for w in windows:
        result[f"corr_{w}d"] = combined["lai"].rolling(w).corr(combined["ret"])
    return pd.DataFrame(result, index=combined.index)


def run_full_backtest():
    """Run the complete backtest suite."""
    print("=" * 70)
    print("LEETCODE ANXIETY INDEX — BACKTEST REPORT")
    print("=" * 70)

    lai_df = load_lai()
    stocks = load_stock_prices()

    lai = lai_df["lai_smoothed"].dropna()
    benchmark_prices = stocks[BENCHMARK].dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Align
    common = lai.index.intersection(benchmark_returns.index)
    lai_aligned = lai.loc[common]
    ret_aligned = benchmark_returns.loc[common]

    print(f"\nData: {len(common)} overlapping trading days")
    print(f"Period: {common.min().strftime('%Y-%m-%d')} to {common.max().strftime('%Y-%m-%d')}")
    print(f"Benchmark: {BENCHMARK}")

    # --- 1. Lead-Lag Correlation ---
    print("\n" + "-" * 50)
    print("1. LEAD-LAG CORRELATION (LAI vs QQQ daily returns)")
    print("-" * 50)
    ll = lead_lag_correlation(lai_aligned, ret_aligned, max_lag=60)
    key_lags = [0, 1, 5, 10, 20, 40, 60]
    for lag in key_lags:
        if lag in ll.index:
            print(f"  Lag {lag:>3d}d: corr = {ll[lag]:+.4f}")

    # Find peak predictive lag
    positive_lags = ll[ll.index > 0]
    if not positive_lags.empty:
        best_lag = positive_lags.abs().idxmax()
        print(f"  Peak predictive lag: {best_lag}d (corr = {ll[best_lag]:+.4f})")

    # --- 2. Granger Causality ---
    print("\n" + "-" * 50)
    print("2. GRANGER CAUSALITY TEST (does LAI predict QQQ returns?)")
    print("-" * 50)
    gc = granger_causality_test(lai_aligned, ret_aligned, max_lag=20)
    if "error" not in gc:
        for lag_num in [1, 5, 10, 15, 20]:
            if lag_num in gc:
                sig = "***" if gc[lag_num] < 0.01 else "**" if gc[lag_num] < 0.05 else "*" if gc[lag_num] < 0.1 else ""
                print(f"  Lag {lag_num:>2d}: p-value = {gc[lag_num]:.6f} {sig}")
    else:
        print(f"  Error: {gc['error']}")

    # --- 3. Regime Analysis ---
    print("\n" + "-" * 50)
    print("3. REGIME ANALYSIS (stock returns by anxiety level)")
    print("-" * 50)
    ra = regime_analysis(lai_aligned, ret_aligned)
    for k, v in ra.items():
        print(f"  {k}: {v}")

    # --- 4. Forward Return Analysis ---
    print("\n" + "-" * 50)
    print(f"4. FORWARD RETURNS AFTER LAI > {THRESHOLD_HIGH}")
    print("-" * 50)
    fr = forward_return_analysis(lai, benchmark_prices)
    for period, metrics in fr.items():
        print(f"\n  {period} forward:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    # --- 5. Event Study ---
    print("\n" + "-" * 50)
    print("5. EVENT STUDY (known tech layoffs)")
    print("-" * 50)
    es = event_study(lai, benchmark_prices)
    for label, data in es.items():
        print(f"\n  {label} ({data['date']}):")
        for k, v in data.items():
            if k != "date":
                print(f"    {k}: {v}")

    # --- 6. Per-Stock Correlation ---
    print("\n" + "-" * 50)
    print("6. PER-STOCK LEAD-LAG CORRELATION (lag 20d)")
    print("-" * 50)
    for ticker in sorted(TICKERS.keys()):
        if ticker in stocks.columns:
            t_ret = stocks[ticker].pct_change().dropna()
            tc = lai.index.intersection(t_ret.index)
            if len(tc) > 60:
                lai_t = lai.loc[tc]
                ret_t = t_ret.loc[tc]
                shifted = ret_t.shift(-20)
                valid = ~(lai_t.isna() | shifted.isna())
                if valid.sum() > 30:
                    corr = lai_t[valid].corr(shifted[valid])
                    print(f"  {ticker:>5s}: corr(LAI, {ticker} 20d fwd ret) = {corr:+.4f}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return {
        "lead_lag": ll,
        "granger": gc,
        "regime": ra,
        "forward_returns": fr,
        "event_study": es,
    }


if __name__ == "__main__":
    run_full_backtest()
