"""Backtesting engine for the LeetCode Anxiety Index.

All results are computed dynamically from real data in the database.
Nothing is hardcoded. This module is re-run by the daily GitHub Actions
pipeline to keep all statistics current.
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BENCHMARK, THRESHOLD_HIGH, THRESHOLD_LOW, LAYOFF_EVENTS, TICKERS
from database.db import get_connection, init_db
from indicator.lai_calculator import load_stock_prices


def load_lai():
    """Load LAI time series."""
    conn = get_connection()
    df = pd.read_sql("SELECT date, lai_smoothed, lai_raw, trends_component, contest_component FROM lai_values", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


# ==============================================================
# CORE ANALYSIS FUNCTIONS
# ==============================================================

def _compute_forward_returns_for_dates(signal_dates, stock_prices, period):
    """Helper: compute forward returns for a list of signal dates."""
    fwd_returns = []
    dates_used = []
    for date in signal_dates:
        future = date + pd.Timedelta(days=period)
        mask = stock_prices.index >= future
        if mask.any():
            future_date = stock_prices.index[mask][0]
            mask_now = stock_prices.index >= date
            if mask_now.any():
                now_date = stock_prices.index[mask_now][0]
                ret = stock_prices.loc[future_date] / stock_prices.loc[now_date] - 1
                fwd_returns.append(float(ret))
                dates_used.append(date)
    return fwd_returns, dates_used


def lead_lag_correlation(lai, stock_returns, max_lag=60):
    """Compute correlation between LAI(t) and stock_return(t+lag)."""
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
        t_stat, p_value = sp_stats.ttest_ind(high, low, equal_var=False)
        result["t_statistic"] = round(t_stat, 4)
        result["p_value"] = round(p_value, 6)
    return result


def forward_return_analysis(lai, stock_prices, threshold=THRESHOLD_HIGH, periods=[5, 10, 20, 60]):
    """After LAI crosses above threshold, compute forward returns."""
    above = lai > threshold
    crossings = above & ~above.shift(1, fill_value=False)
    signal_dates = crossings[crossings].index

    results = {}
    for period in periods:
        fwd_returns, dates_used = _compute_forward_returns_for_dates(signal_dates, stock_prices, period)
        if fwd_returns:
            results[f"{period}d"] = {
                "n_signals": len(fwd_returns),
                "mean_return_pct": round(np.mean(fwd_returns) * 100, 2),
                "median_return_pct": round(np.median(fwd_returns) * 100, 2),
                "win_rate_pct": round(np.mean([r > 0 for r in fwd_returns]) * 100, 1),
                "std_pct": round(np.std(fwd_returns) * 100, 2),
                "sharpe": round(np.mean(fwd_returns) / np.std(fwd_returns), 2) if np.std(fwd_returns) > 0 else 0,
                "max_gain_pct": round(max(fwd_returns) * 100, 2),
                "max_loss_pct": round(min(fwd_returns) * 100, 2),
                "signal_dates": [d.strftime("%Y-%m-%d") for d in dates_used],
                "returns": [round(r * 100, 2) for r in fwd_returns],
            }
    return results


# ==============================================================
# RISK METRICS
# ==============================================================

def compute_risk_metrics(lai, stock_prices, threshold=THRESHOLD_HIGH, holding_period=20):
    """Compute comprehensive risk metrics for the LAI strategy.

    Simulates: buy QQQ when LAI crosses above threshold, hold for holding_period days.
    Compare vs buy-and-hold.
    """
    above = lai > threshold
    crossings = above & ~above.shift(1, fill_value=False)
    signal_dates = crossings[crossings].index

    fwd_returns, dates_used = _compute_forward_returns_for_dates(
        signal_dates, stock_prices, holding_period
    )

    if not fwd_returns:
        return {"error": "no signals"}

    returns_arr = np.array(fwd_returns)
    n = len(returns_arr)

    # --- Basic stats ---
    total_return = np.prod(1 + returns_arr) - 1
    mean_ret = np.mean(returns_arr)
    std_ret = np.std(returns_arr, ddof=1)

    # --- Drawdown analysis ---
    # Simulate equity curve (start at 1.0, apply each trade sequentially)
    equity = [1.0]
    for r in returns_arr:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    # Find worst drawdown period
    dd_end_idx = np.argmin(drawdowns)
    dd_start_idx = np.argmax(equity[:dd_end_idx + 1]) if dd_end_idx > 0 else 0

    # --- VaR and CVaR ---
    var_95 = float(np.percentile(returns_arr, 5))  # 5th percentile = 95% VaR
    var_99 = float(np.percentile(returns_arr, 1))  # 1st percentile = 99% VaR
    cvar_95 = float(returns_arr[returns_arr <= var_95].mean()) if (returns_arr <= var_95).any() else var_95

    # --- Consecutive losses ---
    wins_losses = [1 if r > 0 else 0 for r in returns_arr]
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for wl in wins_losses:
        if wl == 1:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # --- Profit factor ---
    gross_profits = sum(r for r in returns_arr if r > 0)
    gross_losses = abs(sum(r for r in returns_arr if r < 0))
    profit_factor = round(gross_profits / gross_losses, 2) if gross_losses > 0 else float('inf')

    # --- Calmar ratio ---
    # Annualized return based on ACTUAL signal frequency, not theoretical
    # Use real elapsed time between first and last signal to get true trades/year
    first_date = pd.Timestamp(dates_used[0])
    last_date = pd.Timestamp(dates_used[-1])
    elapsed_years = (last_date - first_date).days / 365.25
    if elapsed_years > 0 and n > 1:
        trades_per_year = n / elapsed_years
    else:
        trades_per_year = n  # fallback
    annualized_return = (1 + mean_ret) ** trades_per_year - 1
    calmar = round(annualized_return / abs(max_drawdown), 2) if max_drawdown != 0 else float('inf')

    # --- Buy and hold comparison ---
    bh_start = stock_prices.loc[stock_prices.index >= dates_used[0]].iloc[0]
    bh_end = stock_prices.iloc[-1]
    bh_return = float(bh_end / bh_start - 1)
    bh_days = (stock_prices.index[-1] - pd.Timestamp(dates_used[0])).days
    bh_annualized = (1 + bh_return) ** (365 / bh_days) - 1 if bh_days > 0 else 0

    # Buy and hold drawdown
    bh_prices = stock_prices.loc[stock_prices.index >= dates_used[0]]
    bh_running_max = bh_prices.cummax()
    bh_dd = ((bh_prices - bh_running_max) / bh_running_max).min()

    # --- Sortino ratio ---
    downside_returns = returns_arr[returns_arr < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_ret
    sortino = round(mean_ret / downside_std, 2) if downside_std > 0 else 0

    return {
        "strategy": {
            "total_trades": n,
            "holding_period_days": holding_period,
            "total_return_pct": round(total_return * 100, 2),
            "mean_return_per_trade_pct": round(mean_ret * 100, 2),
            "std_per_trade_pct": round(std_ret * 100, 2),
            "win_rate_pct": round(np.mean(returns_arr > 0) * 100, 1),
            "profit_factor": profit_factor,
            "sharpe_ratio": round(mean_ret / std_ret, 2) if std_ret > 0 else 0,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "max_drawdown_trade_start": dates_used[dd_start_idx] if dd_start_idx < len(dates_used) else "N/A",
            "max_drawdown_trade_end": dates_used[dd_end_idx - 1] if dd_end_idx > 0 and dd_end_idx - 1 < len(dates_used) else "N/A",
            "var_95_pct": round(var_95 * 100, 2),
            "var_99_pct": round(var_99 * 100, 2),
            "cvar_95_pct": round(cvar_95 * 100, 2),
            "max_gain_pct": round(float(returns_arr.max()) * 100, 2),
            "max_loss_pct": round(float(returns_arr.min()) * 100, 2),
            "max_consecutive_wins": max_consec_wins,
            "max_consecutive_losses": max_consec_losses,
            "annualized_return_pct": round(annualized_return * 100, 2),
            "first_signal_date": dates_used[0].strftime("%Y-%m-%d") if hasattr(dates_used[0], 'strftime') else str(dates_used[0]),
            "last_signal_date": dates_used[-1].strftime("%Y-%m-%d") if hasattr(dates_used[-1], 'strftime') else str(dates_used[-1]),
        },
        "buy_and_hold": {
            "total_return_pct": round(bh_return * 100, 2),
            "annualized_return_pct": round(bh_annualized * 100, 2),
            "max_drawdown_pct": round(float(bh_dd) * 100, 2),
        },
        "trade_log": [
            {"date": d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d),
             "return_pct": round(r * 100, 2)}
            for d, r in zip(dates_used, fwd_returns)
        ],
        "equity_curve": [round(e, 4) for e in equity],
    }


# ==============================================================
# MONTE CARLO RANDOM BASELINE
# ==============================================================

def random_baseline(stock_prices, n_signals, periods=[5, 10, 20, 60], n_simulations=10000):
    """Monte Carlo: randomly pick n_signals dates, compute forward returns."""
    trading_days = stock_prices.index
    max_period = max(periods)
    eligible_days = trading_days[:-max_period - 30]

    results = {}
    for period in periods:
        sim_means = []
        sim_win_rates = []
        for _ in range(n_simulations):
            random_dates = np.random.choice(eligible_days, size=n_signals, replace=False)
            fwd_returns, _ = _compute_forward_returns_for_dates(
                pd.DatetimeIndex(random_dates), stock_prices, period
            )
            if fwd_returns:
                sim_means.append(np.mean(fwd_returns) * 100)
                sim_win_rates.append(np.mean([r > 0 for r in fwd_returns]) * 100)

        results[f"{period}d"] = {
            "baseline_mean_return_pct": round(np.mean(sim_means), 2),
            "baseline_win_rate_pct": round(np.mean(sim_win_rates), 1),
            "baseline_std_of_means_pct": round(np.std(sim_means), 2),
            "sim_means": sim_means,
            "sim_win_rates": sim_win_rates,
        }
    return results


def compare_vs_random(lai, stock_prices, threshold=THRESHOLD_HIGH,
                      periods=[5, 10, 20, 60], n_simulations=10000):
    """Compare LAI signal vs random buy baseline with p-values."""
    above = lai > threshold
    crossings = above & ~above.shift(1, fill_value=False)
    signal_dates = crossings[crossings].index
    n_signals = len(signal_dates)

    lai_results = forward_return_analysis(lai, stock_prices, threshold, periods)
    rand_results = random_baseline(stock_prices, n_signals, periods, n_simulations)

    comparison = {}
    for period in periods:
        key = f"{period}d"
        if key not in lai_results or key not in rand_results:
            continue

        lai_mean = lai_results[key]["mean_return_pct"]
        lai_wr = lai_results[key]["win_rate_pct"]
        rand_means = rand_results[key]["sim_means"]
        rand_wrs = rand_results[key]["sim_win_rates"]

        p_value_return = np.mean([m >= lai_mean for m in rand_means])
        p_value_winrate = np.mean([w >= lai_wr for w in rand_wrs])
        percentile_return = round(np.mean([m < lai_mean for m in rand_means]) * 100, 1)
        percentile_winrate = round(np.mean([w < lai_wr for w in rand_wrs]) * 100, 1)

        comparison[key] = {
            "lai_mean_return_pct": lai_mean,
            "lai_win_rate_pct": lai_wr,
            "lai_sharpe": lai_results[key]["sharpe"],
            "random_mean_return_pct": rand_results[key]["baseline_mean_return_pct"],
            "random_win_rate_pct": rand_results[key]["baseline_win_rate_pct"],
            "excess_return_pct": round(lai_mean - rand_results[key]["baseline_mean_return_pct"], 2),
            "excess_win_rate_pct": round(lai_wr - rand_results[key]["baseline_win_rate_pct"], 1),
            "p_value_return": round(p_value_return, 4),
            "p_value_winrate": round(p_value_winrate, 4),
            "percentile_return": percentile_return,
            "percentile_winrate": percentile_winrate,
            "n_signals": n_signals,
        }
    return comparison


def event_study(lai, stock_prices, events=None, window=60):
    """Event study around known layoff dates."""
    if events is None:
        events = LAYOFF_EVENTS
    results = {}
    for date_str, label in events.items():
        event_date = pd.Timestamp(date_str)
        mask = stock_prices.index >= event_date
        if not mask.any():
            continue
        td = stock_prices.index[mask][0]
        future_mask = stock_prices.index >= td
        future_prices = stock_prices.loc[future_mask].head(window + 1)
        if len(future_prices) < 2:
            continue
        cum_return = future_prices.iloc[-1] / future_prices.iloc[0] - 1
        results[label] = {
            "date": date_str,
            "lai_at_event": round(float(lai.loc[td]), 1) if td in lai.index else None,
            f"stock_{window}d_return_pct": round(float(cum_return) * 100, 2),
        }
    return results


# ==============================================================
# SAVE RESULTS TO JSON (for dashboard to read)
# ==============================================================

def save_backtest_results(results, path=None):
    """Save backtest results as JSON for the dashboard to consume."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "backtest_results.json")

    # Clean numpy types for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    cleaned = clean(results)
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Backtest results saved to {path}")


# ==============================================================
# MAIN BACKTEST
# ==============================================================

def run_full_backtest():
    """Run the complete backtest suite and save results."""
    print("=" * 70)
    print("LEETCODE ANXIETY INDEX — BACKTEST REPORT")
    print("=" * 70)

    lai_df = load_lai()
    stocks = load_stock_prices()

    lai = lai_df["lai_smoothed"].dropna()
    benchmark_prices = stocks[BENCHMARK].dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()

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
            if k not in ("signal_dates", "returns"):
                print(f"    {k}: {v}")

    # --- 5. Risk Metrics ---
    print("\n" + "-" * 50)
    print("5. RISK METRICS (20-day holding strategy)")
    print("-" * 50)
    risk = compute_risk_metrics(lai, benchmark_prices, holding_period=20)
    if "error" not in risk:
        print("\n  Strategy Performance:")
        for k, v in risk["strategy"].items():
            if k not in ("first_signal_date", "last_signal_date"):
                print(f"    {k}: {v}")
        print(f"\n  Buy & Hold ({BENCHMARK}) Comparison:")
        for k, v in risk["buy_and_hold"].items():
            print(f"    {k}: {v}")
        print(f"\n  Trade Log:")
        for trade in risk["trade_log"]:
            marker = "+" if trade["return_pct"] > 0 else ""
            print(f"    {trade['date']}: {marker}{trade['return_pct']}%")

    # --- 6. LAI Signal vs Random Buy ---
    print("\n" + "-" * 50)
    print("6. LAI SIGNAL vs RANDOM BUY (10,000 Monte Carlo simulations)")
    print("-" * 50)
    comparison = compare_vs_random(lai, benchmark_prices)
    for period_key, metrics in comparison.items():
        print(f"\n  {period_key} holding period ({metrics['n_signals']} signals):")
        print(f"    LAI Signal:  mean={metrics['lai_mean_return_pct']:+.2f}%  win_rate={metrics['lai_win_rate_pct']:.1f}%  sharpe={metrics['lai_sharpe']}")
        print(f"    Random Buy:  mean={metrics['random_mean_return_pct']:+.2f}%  win_rate={metrics['random_win_rate_pct']:.1f}%")
        print(f"    Excess Return:   {metrics['excess_return_pct']:+.2f}%")
        print(f"    Excess Win Rate: {metrics['excess_win_rate_pct']:+.1f}%")
        print(f"    P-value (return):   {metrics['p_value_return']:.4f}{'  **' if metrics['p_value_return'] < 0.05 else '  *' if metrics['p_value_return'] < 0.1 else ''}")
        print(f"    P-value (win rate): {metrics['p_value_winrate']:.4f}{'  **' if metrics['p_value_winrate'] < 0.05 else '  *' if metrics['p_value_winrate'] < 0.1 else ''}")
        print(f"    Percentile rank:    {metrics['percentile_return']:.1f}th (return), {metrics['percentile_winrate']:.1f}th (win rate)")

    # --- 7. Event Study ---
    print("\n" + "-" * 50)
    print("7. EVENT STUDY (known tech layoffs)")
    print("-" * 50)
    es = event_study(lai, benchmark_prices)
    for label, data in es.items():
        print(f"\n  {label} ({data['date']}):")
        for k, v in data.items():
            if k != "date":
                print(f"    {k}: {v}")

    # --- 8. Per-Stock Correlation ---
    print("\n" + "-" * 50)
    print("8. PER-STOCK LEAD-LAG CORRELATION (lag 20d)")
    print("-" * 50)
    per_stock = {}
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
                    per_stock[ticker] = round(float(corr), 4)
                    print(f"  {ticker:>5s}: corr(LAI, {ticker} 20d fwd ret) = {corr:+.4f}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    # --- Save all results to JSON for dashboard ---
    all_results = {
        "metadata": {
            "data_points": len(common),
            "period_start": common.min().strftime("%Y-%m-%d"),
            "period_end": common.max().strftime("%Y-%m-%d"),
            "benchmark": BENCHMARK,
            "threshold": THRESHOLD_HIGH,
            "computed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "forward_returns": fr,
        "risk_metrics": risk if "error" not in risk else {},
        "comparison_vs_random": {k: {kk: vv for kk, vv in v.items() if kk not in ("sim_means", "sim_win_rates")}
                                  for k, v in comparison.items()},
        "regime_analysis": ra,
        "granger_causality": gc,
        "per_stock_correlation": per_stock,
        "event_study": es,
    }
    save_backtest_results(all_results)

    return all_results


if __name__ == "__main__":
    run_full_backtest()
