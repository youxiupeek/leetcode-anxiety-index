"""LeetCode Anxiety Index — Dash Dashboard.

All backtest/risk data is read from data/backtest_results.json,
which is regenerated daily by GitHub Actions. Nothing is hardcoded.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input

from database.db import get_connection, init_db
from dashboard.charts import (
    build_lai_timeseries,
    build_component_breakdown,
    build_correlation_heatmap,
    build_contest_chart,
    build_scatter_plot,
    build_stats_card,
    DARK_BG, CARD_BG, TEXT_COLOR, GREEN, YELLOW, RED, BLUE, PURPLE,
)
from config.settings import BENCHMARK

# --- Load data ---
init_db()
conn = get_connection()
lai_df = pd.read_sql("SELECT * FROM lai_values", conn)
stock_df = pd.read_sql("SELECT * FROM stock_prices", conn)
contest_df = pd.read_sql("SELECT * FROM contest_participation ORDER BY contest_date", conn)
conn.close()

lai_valid = lai_df.dropna(subset=["lai_smoothed"])
lai_series = lai_valid.set_index(pd.to_datetime(lai_valid["date"]))["lai_smoothed"]

qqq = stock_df[stock_df["ticker"] == BENCHMARK].copy()
qqq["date"] = pd.to_datetime(qqq["date"])
qqq_prices = qqq.set_index("date")["adj_close"].sort_index()

stats = build_stats_card(lai_df, stock_df)

# --- Load backtest results from JSON (dynamically computed) ---
BACKTEST_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "backtest_results.json")
bt = {}
if os.path.exists(BACKTEST_PATH):
    with open(BACKTEST_PATH) as f:
        bt = json.load(f)


# --- Helper components ---

def _stats_grid(stats_data):
    items = [
        ("Current LAI", f"{stats_data['current_lai']}", BLUE),
        ("30d Trend", f"{stats_data['trend_30d']:+.1f}", GREEN if stats_data["trend_30d"] < 0 else RED),
        ("90d Corr w/ QQQ", f"{stats_data['correlation']:.4f}", PURPLE),
        ("Signal", stats_data["signal"],
         RED if stats_data["signal"] == "ELEVATED ANXIETY"
         else GREEN if stats_data["signal"] == "LOW ANXIETY" else YELLOW),
    ]
    cards = []
    for label, value, color in items:
        cards.append(html.Div(
            style={"textAlign": "center", "padding": "12px", "borderRadius": "6px", "backgroundColor": DARK_BG},
            children=[
                html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": color}),
                html.Div(label, style={"fontSize": "11px", "color": "#8b949e", "marginTop": "4px"}),
            ],
        ))
    return html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px"}, children=cards)


def _forward_return_table():
    """Build forward return table from live backtest JSON."""
    fr = bt.get("forward_returns", {})
    if not fr:
        return html.P("Backtest not yet computed", style={"color": "#8b949e", "fontSize": "12px"})

    header = html.Tr([
        html.Th("Period", style={"textAlign": "left", "padding": "4px 8px", "fontSize": "12px", "color": "#8b949e"}),
        html.Th("Mean Ret", style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px", "color": "#8b949e"}),
        html.Th("Win Rate", style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px", "color": "#8b949e"}),
        html.Th("vs Random", style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px", "color": "#8b949e"}),
        html.Th("P-val", style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px", "color": "#8b949e"}),
    ])

    comp = bt.get("comparison_vs_random", {})
    body = []
    for period_key in ["5d", "10d", "20d", "60d"]:
        if period_key not in fr:
            continue
        d = fr[period_key]
        c = comp.get(period_key, {})
        ret = f"+{d['mean_return_pct']:.2f}%" if d["mean_return_pct"] > 0 else f"{d['mean_return_pct']:.2f}%"
        wr = f"{d['win_rate_pct']:.1f}%"
        excess = f"+{c.get('excess_return_pct', 0):.2f}%" if c.get("excess_return_pct", 0) > 0 else f"{c.get('excess_return_pct', 0):.2f}%"
        pval = c.get("p_value_return", 1)
        pval_str = f"{pval:.3f}"
        pval_color = GREEN if pval < 0.05 else YELLOW if pval < 0.1 else "#8b949e"

        body.append(html.Tr([
            html.Td(period_key, style={"padding": "4px 8px", "fontSize": "12px"}),
            html.Td(ret, style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px",
                                "color": GREEN if d["mean_return_pct"] > 0 else RED}),
            html.Td(wr, style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px",
                                "color": GREEN if d["win_rate_pct"] > 50 else RED}),
            html.Td(excess, style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px",
                                   "color": GREEN if c.get("excess_return_pct", 0) > 0 else RED}),
            html.Td(pval_str, style={"textAlign": "right", "padding": "4px 8px", "fontSize": "12px",
                                     "color": pval_color}),
        ]))

    return html.Table(style={"width": "100%", "borderCollapse": "collapse"},
                      children=[html.Thead(header), html.Tbody(body)])


def _risk_metrics_panel():
    """Build risk metrics panel from live backtest JSON."""
    risk = bt.get("risk_metrics", {})
    if not risk or "strategy" not in risk:
        return html.P("Risk metrics not yet computed", style={"color": "#8b949e", "fontSize": "12px"})

    s = risk["strategy"]
    bh = risk["buy_and_hold"]

    def _row(label, strat_val, bh_val=None, fmt="{}", is_pct=False, invert_color=False):
        sv = fmt.format(strat_val) + ("%" if is_pct else "")
        better = strat_val > bh_val if bh_val is not None and not invert_color else strat_val < bh_val if bh_val is not None else True
        cells = [
            html.Td(label, style={"padding": "3px 6px", "fontSize": "11px", "color": "#8b949e"}),
            html.Td(sv, style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px",
                                "fontWeight": "600", "color": GREEN if better else RED}),
        ]
        if bh_val is not None:
            bv = fmt.format(bh_val) + ("%" if is_pct else "")
            cells.append(html.Td(bv, style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px", "color": TEXT_COLOR}))
        return html.Tr(cells)

    header = html.Tr([
        html.Th("", style={"padding": "3px 6px", "fontSize": "11px"}),
        html.Th("LAI Strategy", style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px", "color": BLUE}),
        html.Th("Buy & Hold", style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px", "color": PURPLE}),
    ])

    rows = [
        _row("Annualized Return", s["annualized_return_pct"], bh["annualized_return_pct"], "{:.1f}", True),
        _row("Total Return", s["total_return_pct"], bh["total_return_pct"], "{:.1f}", True),
        _row("Max Drawdown", s["max_drawdown_pct"], bh["max_drawdown_pct"], "{:.1f}", True, invert_color=True),
        _row("Sharpe Ratio", s["sharpe_ratio"], None, "{:.2f}"),
        _row("Sortino Ratio", s["sortino_ratio"], None, "{:.2f}"),
        _row("Calmar Ratio", s["calmar_ratio"], None, "{:.2f}"),
        _row("Profit Factor", s["profit_factor"], None, "{:.2f}"),
        _row("VaR 95%", s["var_95_pct"], None, "{:.2f}", True),
        _row("CVaR 95%", s["cvar_95_pct"], None, "{:.2f}", True),
        _row("Win Rate", s["win_rate_pct"], None, "{:.1f}", True),
        _row("Max Consec Losses", s["max_consecutive_losses"], None, "{}"),
        _row("Worst Trade", s["max_loss_pct"], None, "{:.2f}", True),
        _row("Best Trade", s["max_gain_pct"], None, "{:.2f}", True),
    ]

    meta = bt.get("metadata", {})
    computed_at = meta.get("computed_at", "unknown")

    return html.Div([
        html.Table(style={"width": "100%", "borderCollapse": "collapse"},
                   children=[html.Thead(header), html.Tbody(rows)]),
        html.P(f"Based on {s['total_trades']} trades, {s['holding_period_days']}d hold | Updated: {computed_at}",
               style={"fontSize": "10px", "color": "#484f58", "marginTop": "8px"}),
    ])


def _trade_log_table():
    """Show individual trade log from backtest."""
    risk = bt.get("risk_metrics", {})
    if not risk or "trade_log" not in risk:
        return html.P("No trade log", style={"color": "#8b949e", "fontSize": "12px"})

    header = html.Tr([
        html.Th("#", style={"padding": "3px 6px", "fontSize": "11px", "color": "#8b949e"}),
        html.Th("Entry Date", style={"padding": "3px 6px", "fontSize": "11px", "color": "#8b949e"}),
        html.Th("Return", style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px", "color": "#8b949e"}),
    ])

    rows = []
    for i, trade in enumerate(risk["trade_log"], 1):
        ret = trade["return_pct"]
        rows.append(html.Tr([
            html.Td(str(i), style={"padding": "3px 6px", "fontSize": "11px", "color": "#8b949e"}),
            html.Td(trade["date"], style={"padding": "3px 6px", "fontSize": "11px"}),
            html.Td(f"{ret:+.2f}%", style={"textAlign": "right", "padding": "3px 6px", "fontSize": "11px",
                                             "fontWeight": "600", "color": GREEN if ret > 0 else RED}),
        ]))

    return html.Table(style={"width": "100%", "borderCollapse": "collapse"},
                      children=[html.Thead(header), html.Tbody(rows)])


# --- App ---
app = Dash(__name__, title="LeetCode Anxiety Index")
server = app.server

app.layout = html.Div(
    style={"backgroundColor": DARK_BG, "color": TEXT_COLOR, "fontFamily": "'Inter', 'Segoe UI', sans-serif",
           "minHeight": "100vh", "padding": "20px", "maxWidth": "1400px", "margin": "0 auto"},
    children=[
        # Header
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                   "marginBottom": "20px", "borderBottom": f"1px solid {CARD_BG}", "paddingBottom": "15px"},
            children=[
                html.Div([
                    html.H1("LeetCode Anxiety Index", style={"margin": "0", "fontSize": "28px", "fontWeight": "700"}),
                    html.P("Tracking tech layoff anxiety through coding interview preparation activity",
                           style={"margin": "4px 0 0 0", "color": "#8b949e", "fontSize": "14px"}),
                ]),
                html.Div(
                    style={"display": "flex", "gap": "15px", "alignItems": "center"},
                    children=[
                        html.Div(
                            style={"textAlign": "center", "padding": "10px 20px", "borderRadius": "8px",
                                   "backgroundColor": RED if stats["signal"] == "ELEVATED ANXIETY"
                                   else GREEN if stats["signal"] == "LOW ANXIETY" else YELLOW,
                                   "color": "#fff" if stats["signal"] != "NORMAL" else "#000"},
                            children=[
                                html.Div(f"{stats['current_lai']}", style={"fontSize": "32px", "fontWeight": "700"}),
                                html.Div(stats["signal"], style={"fontSize": "11px", "fontWeight": "600"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Panel 1: LAI Time Series
        html.Div(style={"marginBottom": "20px"},
                 children=[dcc.Graph(figure=build_lai_timeseries(lai_df, stock_df), config={"displayModeBar": False})]),

        # Row 2: Components + Correlation Heatmap
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div(style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "5px"},
                         children=[dcc.Graph(figure=build_component_breakdown(lai_df), config={"displayModeBar": False})]),
                html.Div(style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "5px"},
                         children=[dcc.Graph(figure=build_correlation_heatmap(lai_series, stock_df), config={"displayModeBar": False})]),
            ],
        ),

        # Row 3: Risk Metrics + Stats/Forward Returns
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                # Risk Metrics Panel (NEW)
                html.Div(
                    style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "20px"},
                    children=[
                        html.H3("Risk Metrics: LAI Strategy vs Buy & Hold",
                                style={"margin": "0 0 15px 0", "fontSize": "15px"}),
                        _risk_metrics_panel(),
                    ],
                ),
                # Stats + Forward Returns
                html.Div(
                    style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "20px"},
                    children=[
                        html.H3("Signal Summary", style={"margin": "0 0 15px 0", "fontSize": "15px"}),
                        _stats_grid(stats),
                        html.Hr(style={"borderColor": DARK_BG, "margin": "15px 0"}),
                        html.H4("Forward Returns After LAI > 70 (vs Random Buy)",
                                style={"margin": "0 0 8px 0", "fontSize": "13px"}),
                        _forward_return_table(),
                    ],
                ),
            ],
        ),

        # Row 4: Contest Chart + Trade Log
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div(style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "5px"},
                         children=[dcc.Graph(figure=build_contest_chart(contest_df), config={"displayModeBar": False})]),
                html.Div(
                    style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "20px"},
                    children=[
                        html.H3("Trade Log (Every LAI > 70 Signal)",
                                style={"margin": "0 0 15px 0", "fontSize": "15px"}),
                        _trade_log_table(),
                    ],
                ),
            ],
        ),

        # Row 5: Scatter with dropdown
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.Div(
                    style={"backgroundColor": CARD_BG, "borderRadius": "8px", "padding": "5px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "10px", "padding": "10px 15px"},
                            children=[
                                html.Label("Forward Period:", style={"fontSize": "13px"}),
                                dcc.Dropdown(
                                    id="fwd-period-dropdown",
                                    options=[{"label": f"{d} days", "value": d} for d in [5, 10, 20, 60]],
                                    value=20,
                                    style={"width": "120px", "backgroundColor": DARK_BG, "color": TEXT_COLOR},
                                    clearable=False,
                                ),
                            ],
                        ),
                        dcc.Graph(id="scatter-plot", config={"displayModeBar": False}),
                    ],
                ),
            ],
        ),

        # Disclaimers
        html.Div(
            style={"backgroundColor": "#1c1c1c", "borderRadius": "8px", "padding": "15px", "marginBottom": "20px",
                   "border": f"1px solid {YELLOW}", "fontSize": "11px", "color": YELLOW, "lineHeight": "1.6"},
            children=[
                html.Strong("IMPORTANT DISCLAIMERS:"),
                html.Ul(style={"margin": "8px 0 0 0", "paddingLeft": "20px"}, children=[
                    html.Li("This is a RESEARCH INDICATOR, not financial advice. Past performance does not predict future results."),
                    html.Li(f"Only {bt.get('risk_metrics', {}).get('strategy', {}).get('total_trades', 'N/A')} historical signals — sample size is too small for high-confidence conclusions."),
                    html.Li("Google Trends data is interpolated from weekly to daily — introduces smoothing artifacts."),
                    html.Li("Contest 'registerUserNum' (GraphQL) differs from actual participants for recent contests."),
                    html.Li("The 2020-2022 period was anomalous (COVID, zero rates, meme stocks). Backtest may not generalize."),
                    html.Li("Granger causality ≠ true causation. Correlation can break at any time."),
                    html.Li("VaR/CVaR computed from 19 trades — statistically unreliable, treat as directional only."),
                ]),
            ],
        ),

        # Footer
        html.Div(
            style={"textAlign": "center", "padding": "15px", "color": "#484f58", "fontSize": "11px",
                   "borderTop": f"1px solid {CARD_BG}"},
            children=[
                html.P(f"Data as of {stats['latest_date']} | All metrics computed from real data, updated daily via GitHub Actions"),
                html.P(f"Backtest last computed: {bt.get('metadata', {}).get('computed_at', 'N/A')}"),
            ],
        ),
    ],
)


@callback(Output("scatter-plot", "figure"), Input("fwd-period-dropdown", "value"))
def update_scatter(fwd_days):
    return build_scatter_plot(lai_series, qqq_prices, forward_days=fwd_days)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
