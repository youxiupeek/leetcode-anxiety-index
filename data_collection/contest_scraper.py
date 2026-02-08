"""Scrape LeetCode contest participation data via GraphQL (bypasses Cloudflare).

Uses two sources:
1. Leetcodescraper GitHub repo (actual participant counts, through Sept 2024)
2. LeetCode GraphQL API (registerUserNum, for recent contests)
"""
import sys
import os
import time
import json
import requests
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SCRAPE_DELAY_SECONDS, START_DATE
from database.db import get_connection, init_db, upsert_contest

GRAPHQL_URL = "https://leetcode.com/graphql/"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://leetcode.com/contest/",
}

GITHUB_RAW = "https://raw.githubusercontent.com/Leader-board/Leetcodescraper/master/stats"


def fetch_all_contests_graphql():
    """Get list of all contests with slugs and start times."""
    query = """
    {
        allContests {
            title
            titleSlug
            startTime
        }
    }
    """
    resp = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    contests = data["data"]["allContests"]
    print(f"Found {len(contests)} total contests from GraphQL")
    return contests


def fetch_contest_detail_batch(slugs):
    """Batch-fetch registerUserNum for multiple contests in one GraphQL request."""
    aliases = []
    for i, slug in enumerate(slugs):
        safe_alias = f"c{i}"
        aliases.append(f'{safe_alias}: contestDetailPage(contestSlug: "{slug}") {{ title titleSlug startTime registerUserNum }}')

    query = "{ " + " ".join(aliases) + " }"
    resp = requests.post(GRAPHQL_URL, json={"query": query}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = {}
    for key, val in data.get("data", {}).items():
        if val and "titleSlug" in val:
            results[val["titleSlug"]] = val.get("registerUserNum", 0)
    return results


def fetch_from_github_stats(contest_type, number):
    """Fetch actual participant count from Leetcodescraper GitHub repo."""
    if contest_type == "weekly":
        filename = f"weekly{number}.txt"
    else:
        filename = f"biweekly{number}.txt"

    url = f"{GITHUB_RAW}/{filename}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        # Parse "Number of participants XXXXX"
        for line in resp.text.split("\n"):
            if "Number of participants" in line:
                parts = line.strip().split()
                return int(parts[-1])
    except Exception:
        pass
    return None


def scrape_all_contests():
    """Main: scrape all contest data using hybrid approach."""
    init_db()
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")

    # --- Step 1: Get all contest slugs from GraphQL ---
    print("Step 1: Fetching contest list from GraphQL...")
    all_contests = fetch_all_contests_graphql()

    # Filter to 2020+ and parse
    contests_to_process = []
    for c in all_contests:
        ts = int(c["startTime"])
        dt = datetime.utcfromtimestamp(ts)
        if dt < start_dt:
            continue
        slug = c["titleSlug"]
        if slug.startswith("weekly-contest-"):
            ctype = "weekly"
            number = int(slug.replace("weekly-contest-", ""))
        elif slug.startswith("biweekly-contest-"):
            ctype = "biweekly"
            number = int(slug.replace("biweekly-contest-", ""))
        else:
            continue
        contests_to_process.append({
            "slug": slug,
            "type": ctype,
            "number": number,
            "date": dt.strftime("%Y-%m-%d"),
            "datetime": dt,
        })

    contests_to_process.sort(key=lambda x: x["datetime"])
    print(f"Contests from {START_DATE} onward: {len(contests_to_process)}")

    # --- Step 2: Try GitHub stats first (actual participants, faster) ---
    print("\nStep 2: Fetching from Leetcodescraper GitHub (actual participant counts)...")
    github_success = 0
    github_fail = 0

    for c in contests_to_process:
        count = fetch_from_github_stats(c["type"], c["number"])
        if count is not None:
            c["participant_count"] = count
            c["source"] = "github"
            github_success += 1
        else:
            c["participant_count"] = None
            github_fail += 1

        if (github_success + github_fail) % 50 == 0:
            print(f"  Progress: {github_success + github_fail}/{len(contests_to_process)} "
                  f"(found: {github_success}, missing: {github_fail})")
        time.sleep(0.1)  # gentle on GitHub

    print(f"  GitHub: found {github_success}, missing {github_fail}")

    # --- Step 3: Fill gaps with GraphQL registerUserNum ---
    missing = [c for c in contests_to_process if c["participant_count"] is None]
    if missing:
        print(f"\nStep 3: Fetching {len(missing)} missing contests from GraphQL (batched)...")
        batch_size = 10
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            slugs = [c["slug"] for c in batch]
            try:
                counts = fetch_contest_detail_batch(slugs)
                for c in batch:
                    if c["slug"] in counts:
                        c["participant_count"] = counts[c["slug"]]
                        c["source"] = "graphql"
                print(f"  Batch {i // batch_size + 1}: fetched {len(counts)} contests")
            except Exception as e:
                print(f"  Batch {i // batch_size + 1}: ERROR - {e}")
            time.sleep(SCRAPE_DELAY_SECONDS)
    else:
        print("\nStep 3: No gaps to fill, all data from GitHub.")

    # --- Step 4: Save to database (filter out future/invalid contests) ---
    print("\nStep 4: Saving to database...")
    today = datetime.utcnow()
    saved = 0
    skipped = 0
    for c in contests_to_process:
        if c["participant_count"] is None or c["participant_count"] <= 0:
            continue
        # Skip future contests (pre-registration numbers, not actual participants)
        if c["datetime"] > today:
            skipped += 1
            continue
        # Skip suspiciously low counts (likely incomplete data)
        if c["participant_count"] < 500:
            skipped += 1
            continue
        upsert_contest(c["slug"], c["type"], c["number"], c["date"], c["participant_count"])
        saved += 1

    if skipped:
        print(f"  Skipped {skipped} contests (future dates or <500 participants)")

    print(f"Saved {saved} contests to database")

    # --- Summary ---
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM contest_participation ORDER BY contest_date", conn)
    conn.close()

    if not df.empty:
        print(f"\nSummary:")
        print(f"  Total contests in DB: {len(df)}")
        print(f"  Date range: {df['contest_date'].min()} to {df['contest_date'].max()}")
        print(f"  Participants: min={df['participant_count'].min():,}, "
              f"max={df['participant_count'].max():,}, "
              f"mean={df['participant_count'].mean():,.0f}")

        # Show a few recent
        print(f"\n  Recent contests:")
        for _, row in df.tail(5).iterrows():
            print(f"    {row['contest_slug']}: {row['participant_count']:,} participants ({row['contest_date']})")


if __name__ == "__main__":
    scrape_all_contests()
