"""Email alert system for LAI buy/sell signals.

Checks if LAI has crossed above 70 (BUY) or if 20 trading days have
passed since entry (SELL). Sends email via Gmail SMTP.

Requires env vars: GMAIL_USER, GMAIL_APP_PASSWORD
If not set, prints alerts to stdout instead.
"""
import sys
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import THRESHOLD_HIGH, BENCHMARK
from database.db import get_connection, init_db

ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "yangmingliu16@gmail.com")
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
DASHBOARD_URL = "https://leetcode-anxiety-index.onrender.com/"
HOLDING_PERIOD = 20  # trading days

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "signal_state.json")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"active_signals": [], "last_lai": None, "last_date": None}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def send_email(subject, body_html):
    """Send email via Gmail SMTP. Falls back to stdout if no credentials."""
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        print(f"\n{'='*60}")
        print(f"EMAIL ALERT (no credentials, printing to stdout)")
        print(f"To: {ALERT_EMAIL}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        print(body_html.replace("<br>", "\n").replace("<b>", "").replace("</b>", ""))
        print(f"{'='*60}\n")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = ALERT_EMAIL
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, ALERT_EMAIL, msg.as_string())
        print(f"Email sent: {subject}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


def get_latest_lai():
    """Get the two most recent LAI values (for crossing detection)."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, lai_smoothed FROM lai_values WHERE lai_smoothed IS NOT NULL "
        "ORDER BY date DESC LIMIT 2",
        conn,
    )
    conn.close()
    if len(df) < 2:
        return None, None, None, None
    today_row = df.iloc[0]
    yesterday_row = df.iloc[1]
    return (
        today_row["date"],
        float(today_row["lai_smoothed"]),
        yesterday_row["date"],
        float(yesterday_row["lai_smoothed"]),
    )


def get_qqq_price(date_str):
    """Get QQQ closing price for a given date (or nearest prior)."""
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT adj_close FROM stock_prices WHERE ticker = '{BENCHMARK}' "
        f"AND date <= '{date_str}' ORDER BY date DESC LIMIT 1",
        conn,
    )
    conn.close()
    if df.empty:
        return None
    return float(df.iloc[0]["adj_close"])


def count_trading_days(start_date, end_date):
    """Count trading days between two dates using stock_prices table."""
    conn = get_connection()
    df = pd.read_sql(
        f"SELECT DISTINCT date FROM stock_prices WHERE ticker = '{BENCHMARK}' "
        f"AND date > '{start_date}' AND date <= '{end_date}' ORDER BY date",
        conn,
    )
    conn.close()
    return len(df)


def check_signals():
    """Main: check for buy/sell signals and send alerts."""
    init_db()
    state = load_state()

    today_date, today_lai, yesterday_date, yesterday_lai = get_latest_lai()
    if today_date is None:
        print("No LAI data available, skipping signal check.")
        return

    print(f"Signal check: {today_date}")
    print(f"  LAI today:     {today_lai:.1f}")
    print(f"  LAI yesterday: {yesterday_lai:.1f}")
    print(f"  Threshold:     {THRESHOLD_HIGH}")
    print(f"  Active signals: {len(state['active_signals'])}")

    # --- BUY CHECK ---
    if today_lai > THRESHOLD_HIGH and yesterday_lai <= THRESHOLD_HIGH:
        entry_price = get_qqq_price(today_date)
        print(f"\n  *** BUY SIGNAL DETECTED ***")
        print(f"  LAI crossed above {THRESHOLD_HIGH}: {yesterday_lai:.1f} -> {today_lai:.1f}")

        # Record signal
        signal = {
            "entry_date": today_date,
            "entry_lai": round(today_lai, 1),
            "entry_price": round(entry_price, 2) if entry_price else None,
        }
        state["active_signals"].append(signal)

        # Send email
        subject = f"[LAI BUY] LAI = {today_lai:.1f} -- Buy QQQ now"
        body = f"""
        <h2 style="color: #f85149;">BUY SIGNAL - LeetCode Anxiety Index</h2>
        <p><b>Date:</b> {today_date}</p>
        <p><b>LAI:</b> {today_lai:.1f} (crossed above {THRESHOLD_HIGH})</p>
        <p><b>QQQ Price:</b> ${entry_price:.2f}</p>
        <br>
        <h3>Action:</h3>
        <p>Buy QQQ at market open tomorrow. Hold for <b>20 trading days</b>.</p>
        <br>
        <h3>Historical Stats (19 signals):</h3>
        <ul>
            <li>Win rate: 89.5%</li>
            <li>Average return: +3.81%</li>
            <li>Max drawdown: -5.08%</li>
        </ul>
        <br>
        <p><a href="{DASHBOARD_URL}">View Dashboard</a></p>
        <hr>
        <p style="color: #8b949e; font-size: 11px;">
        This is a research indicator, not financial advice.
        </p>
        """
        send_email(subject, body)

    elif today_lai > THRESHOLD_HIGH:
        print(f"  LAI above {THRESHOLD_HIGH} but not a new crossing (already elevated)")
    else:
        print(f"  LAI below {THRESHOLD_HIGH}, no buy signal")

    # --- SELL CHECK ---
    remaining = []
    for signal in state["active_signals"]:
        entry_date = signal["entry_date"]
        days_held = count_trading_days(entry_date, today_date)
        print(f"\n  Active position: entry {entry_date}, {days_held} trading days held")

        if days_held >= HOLDING_PERIOD:
            current_price = get_qqq_price(today_date)
            entry_price = signal.get("entry_price")
            if entry_price and current_price:
                ret = (current_price / entry_price - 1) * 100
                ret_str = f"{ret:+.2f}%"
            else:
                ret = None
                ret_str = "N/A"

            print(f"  *** SELL SIGNAL *** 20 trading days reached")
            print(f"  Entry: {entry_date} @ ${entry_price}, Current: ${current_price:.2f}, Return: {ret_str}")

            subject = f"[LAI SELL] 20 days reached -- Sell QQQ (entry: {entry_date})"
            body = f"""
            <h2 style="color: #3fb950;">SELL REMINDER - LeetCode Anxiety Index</h2>
            <p><b>Date:</b> {today_date}</p>
            <p><b>Holding period:</b> {days_held} trading days (target: {HOLDING_PERIOD})</p>
            <br>
            <h3>Position:</h3>
            <ul>
                <li>Entry date: {entry_date}</li>
                <li>Entry LAI: {signal.get('entry_lai', 'N/A')}</li>
                <li>Entry QQQ price: ${entry_price:.2f}</li>
                <li>Current QQQ price: ${current_price:.2f}</li>
                <li><b>Return: {ret_str}</b></li>
            </ul>
            <br>
            <h3>Action:</h3>
            <p>Sell QQQ at market open tomorrow.</p>
            <br>
            <p><a href="{DASHBOARD_URL}">View Dashboard</a></p>
            <hr>
            <p style="color: #8b949e; font-size: 11px;">
            This is a research indicator, not financial advice.
            </p>
            """
            send_email(subject, body)
            # Don't keep this signal (position closed)
        else:
            remaining.append(signal)
            print(f"  Holding... {HOLDING_PERIOD - days_held} trading days remaining")

    state["active_signals"] = remaining
    state["last_lai"] = round(today_lai, 1)
    state["last_date"] = today_date
    save_state(state)
    print(f"\nSignal state saved. Active positions: {len(remaining)}")


if __name__ == "__main__":
    check_signals()
