"""Configuration for LeetCode Anxiety Index."""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "lai.db")

# --- Time Range ---
START_DATE = "2020-01-01"

# --- Stock Universe ---
TICKERS = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "QQQ": "Nasdaq 100 ETF",
}

# Primary benchmark for correlation analysis
BENCHMARK = "QQQ"

# --- LAI Weights ---
WEIGHT_TRENDS = 0.6
WEIGHT_CONTEST = 0.4

# --- Normalization ---
ROLLING_WINDOW = 90  # days for z-score normalization
EMA_SPAN = 5         # days for smoothing

# --- Anxiety Thresholds ---
THRESHOLD_LOW = 30
THRESHOLD_HIGH = 70

# --- Contest Scraping ---
WEEKLY_CONTEST_START = 78
WEEKLY_CONTEST_END = 445      # approximate current
BIWEEKLY_CONTEST_START = 1
BIWEEKLY_CONTEST_END = 155    # approximate current
SCRAPE_DELAY_SECONDS = 2      # safe mode

# --- Known Layoff Events (for event study) ---
LAYOFF_EVENTS = {
    "2022-11-09": "Meta 11k layoffs",
    "2023-01-20": "Google 12k layoffs",
    "2023-01-18": "Microsoft 10k layoffs",
    "2023-01-04": "Amazon 18k layoffs",
    "2023-03-20": "Meta 10k layoffs (round 2)",
    "2023-04-19": "Amazon 9k layoffs (round 2)",
    "2024-01-10": "Google hundreds laid off",
    "2024-01-25": "Microsoft 1.9k gaming layoffs",
    "2024-04-02": "Apple EV team layoffs",
}
