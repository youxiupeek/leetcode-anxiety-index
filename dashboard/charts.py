"""Plotly chart builders for the LAI dashboard."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import THRESHOLD_LOW, THRESHOLD_HIGH, LAYOFF_EVENTS, BENCHMARK, TICKERS


DARK_BG = "#0d1117"
CARD_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"


def build_lai_timeseries(lai_df, stock_df):
    """Panel 1: LAI time series with anxiety bands + QQQ overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    dates = pd.to_datetime(lai_df["date"])
    lai = lai_df["lai_smoothed"]

    # Anxiety bands (background)
    fig.add_hrect(y0=0, y1=THRESHOLD_LOW, fillcolor=GREEN, opacity=0.07,
                  line_width=0, layer="below")
    fig.add_hrect(y0=THRESHOLD_LOW, y1=THRESHOLD_HIGH, fillcolor=YELLOW, opacity=0.07,
                  line_width=0, layer="below")
    fig.add_hrect(y0=THRESHOLD_HIGH, y1=100, fillcolor=RED, opacity=0.07,
                  line_width=0, layer="below")

    # LAI line
    colors = []
    for v in lai:
        if pd.isna(v):
            colors.append(TEXT_COLOR)
        elif v >= THRESHOLD_HIGH:
            colors.append(RED)
        elif v <= THRESHOLD_LOW:
            colors.append(GREEN)
        else:
            colors.append(YELLOW)

    fig.add_trace(
        go.Scatter(
            x=dates, y=lai, name="LAI",
            line=dict(color=BLUE, width=2),
            hovertemplate="LAI: %{y:.1f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # QQQ overlay
    qqq = stock_df[stock_df["ticker"] == BENCHMARK].copy()
    qqq["date"] = pd.to_datetime(qqq["date"])
    fig.add_trace(
        go.Scatter(
            x=qqq["date"], y=qqq["adj_close"], name=BENCHMARK,
            line=dict(color=PURPLE, width=1.5, dash="dot"),
            opacity=0.7,
            hovertemplate=f"{BENCHMARK}: $%{{y:.2f}}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Layoff event markers
    for date_str, label in LAYOFF_EVENTS.items():
        event_dt = pd.Timestamp(date_str)
        fig.add_shape(
            type="line", x0=event_dt, x1=event_dt, y0=0, y1=100,
            yref="y", line=dict(color=RED, width=1, dash="dash"), opacity=0.4,
        )
        fig.add_annotation(
            x=event_dt, y=98, yref="y",
            text=label.split(" ")[0], showarrow=False,
            font=dict(size=9, color=RED), textangle=-90,
        )

    fig.update_layout(
        title=dict(text="LeetCode Anxiety Index (LAI)", font=dict(size=20, color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=450,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="LAI (0-100)", range=[0, 100], secondary_y=False,
                     gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text=f"{BENCHMARK} Price ($)", secondary_y=True,
                     gridcolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR)

    return fig


def build_component_breakdown(lai_df):
    """Panel 2: Stacked area of trends vs contest contributions."""
    fig = go.Figure()

    dates = pd.to_datetime(lai_df["date"])

    fig.add_trace(go.Scatter(
        x=dates, y=lai_df["trends_component"], name="Google Trends",
        fill="tozeroy", fillcolor="rgba(88,166,255,0.3)",
        line=dict(color=BLUE, width=1),
        hovertemplate="Trends: %{y:.1f}<extra></extra>",
    ))

    if "contest_component" in lai_df.columns and lai_df["contest_component"].notna().any():
        fig.add_trace(go.Scatter(
            x=dates, y=lai_df["contest_component"], name="Contest Participation",
            fill="tozeroy", fillcolor="rgba(188,140,255,0.3)",
            line=dict(color=PURPLE, width=1),
            hovertemplate="Contest: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="LAI Components (Normalized 0-100)", font=dict(size=14, color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=350,
        margin=dict(l=50, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[0, 100], gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
    )

    return fig


def build_correlation_heatmap(lai_series, stock_df):
    """Panel 3: Heatmap of LAI vs each stock at different lags."""
    lags = [1, 5, 10, 20, 40, 60]
    tickers = sorted([t for t in TICKERS.keys()])

    lai_idx = lai_series.index

    corr_matrix = []
    for ticker in tickers:
        t_data = stock_df[stock_df["ticker"] == ticker].copy()
        t_data["date"] = pd.to_datetime(t_data["date"])
        t_data = t_data.set_index("date").sort_index()
        t_ret = t_data["adj_close"].pct_change().dropna()

        common = lai_idx.intersection(t_ret.index)
        lai_c = lai_series.loc[common]
        ret_c = t_ret.loc[common]

        row = []
        for lag in lags:
            shifted = ret_c.shift(-lag)
            valid = ~(lai_c.isna() | shifted.isna())
            if valid.sum() > 30:
                row.append(round(lai_c[valid].corr(shifted[valid]), 4))
            else:
                row.append(0)
        corr_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=[f"{l}d" for l in lags],
        y=tickers,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-0.1,
        zmax=0.1,
        text=[[f"{v:.3f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Ticker: %{y}<br>Lag: %{x}<br>Corr: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Lead-Lag Correlation: LAI → Stock Returns", font=dict(size=14, color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=350,
        margin=dict(l=60, r=20, t=50, b=30),
        xaxis_title="Forward Lag (days)",
        yaxis_title="",
    )

    return fig


def build_contest_chart(contest_df):
    """Panel 4: Contest participation bar chart."""
    if contest_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Contest data loading...", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=TEXT_COLOR))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=350, title=dict(text="Contest Participation", font=dict(size=14, color=TEXT_COLOR)),
        )
        return fig

    dates = pd.to_datetime(contest_df["contest_date"])
    counts = contest_df["participant_count"]

    colors = [BLUE if t == "weekly" else PURPLE for t in contest_df["contest_type"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=counts, name="Participants",
        marker_color=colors, opacity=0.7,
        hovertemplate="Date: %{x}<br>Participants: %{y:,}<extra></extra>",
    ))

    # Trend line
    if len(counts) > 10:
        z = np.polyfit(range(len(counts)), counts, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=dates, y=p(range(len(counts))),
            name="Trend", line=dict(color=RED, width=2, dash="dash"),
        ))

    fig.update_layout(
        title=dict(text="LeetCode Contest Participation", font=dict(size=14, color=TEXT_COLOR)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=350,
        margin=dict(l=50, r=20, t=50, b=30),
        yaxis=dict(gridcolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        showlegend=False,
    )

    return fig


def build_scatter_plot(lai_series, stock_prices, forward_days=20):
    """Panel 6: Scatter — LAI vs forward N-day return."""
    fwd_ret = stock_prices.pct_change(forward_days).shift(-forward_days)

    common = lai_series.index.intersection(fwd_ret.index)
    x = lai_series.loc[common].dropna()
    y = fwd_ret.loc[common].dropna()
    common2 = x.index.intersection(y.index)
    x = x.loc[common2]
    y = y.loc[common2]

    # Regression
    valid = ~(x.isna() | y.isna())
    if valid.sum() > 10:
        from scipy import stats as sp_stats
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x[valid], y[valid])
        r_sq = r_value ** 2
    else:
        slope, intercept, r_sq, p_value = 0, 0, 0, 1

    fig = go.Figure()

    # Color by LAI regime
    colors = []
    for v in x:
        if v >= THRESHOLD_HIGH:
            colors.append(RED)
        elif v <= THRESHOLD_LOW:
            colors.append(GREEN)
        else:
            colors.append(YELLOW)

    fig.add_trace(go.Scatter(
        x=x, y=y * 100, mode="markers", name="Data",
        marker=dict(color=colors, size=5, opacity=0.5),
        hovertemplate="LAI: %{x:.1f}<br>Fwd Return: %{y:.2f}%<extra></extra>",
    ))

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = (slope * x_line + intercept) * 100
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name=f"R²={r_sq:.4f}",
        line=dict(color=BLUE, width=2, dash="dash"),
    ))

    fig.update_layout(
        title=dict(
            text=f"LAI vs {BENCHMARK} {forward_days}-Day Forward Return (R²={r_sq:.4f}, p={p_value:.4f})",
            font=dict(size=14, color=TEXT_COLOR),
        ),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=350,
        margin=dict(l=50, r=20, t=50, b=30),
        xaxis=dict(title="LAI", gridcolor=GRID_COLOR),
        yaxis=dict(title=f"{forward_days}-Day Forward Return (%)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def build_stats_card(lai_df, stock_df):
    """Panel 5: Stats card with current values."""
    valid = lai_df.dropna(subset=["lai_smoothed"])
    if valid.empty:
        return {"current_lai": 0, "trend_30d": 0, "correlation": 0, "signal": "NO DATA"}

    latest = valid.iloc[-1]
    current = float(latest["lai_smoothed"])

    # 30-day trend
    if len(valid) >= 30:
        lai_30d_ago = float(valid.iloc[-30]["lai_smoothed"])
        trend = current - lai_30d_ago
    else:
        trend = 0

    # Current correlation (90d rolling)
    lai_series = valid.set_index(pd.to_datetime(valid["date"]))["lai_smoothed"]
    qqq = stock_df[stock_df["ticker"] == BENCHMARK].copy()
    qqq["date"] = pd.to_datetime(qqq["date"])
    qqq = qqq.set_index("date").sort_index()
    qqq_ret = qqq["adj_close"].pct_change()

    common = lai_series.index.intersection(qqq_ret.index)
    if len(common) >= 90:
        corr = lai_series.loc[common[-90:]].corr(qqq_ret.loc[common[-90:]])
    else:
        corr = 0

    # Signal
    if current >= THRESHOLD_HIGH:
        signal = "ELEVATED ANXIETY"
    elif current <= THRESHOLD_LOW:
        signal = "LOW ANXIETY"
    else:
        signal = "NORMAL"

    return {
        "current_lai": round(current, 1),
        "trend_30d": round(trend, 1),
        "correlation": round(corr, 4) if not pd.isna(corr) else 0,
        "signal": signal,
        "latest_date": str(latest["date"]),
    }
