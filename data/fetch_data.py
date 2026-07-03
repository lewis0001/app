"""Fetch historical BTC-USD hourly candles from the Coinbase Exchange public API.

Paginates backwards in 300-candle chunks (API limit) and writes a single
CSV sorted by time. No API key required.

Usage:
    python data/fetch_data.py [--start 2021-01-01] [--out data/btc_1h.csv]
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.exchange.coinbase.com/products/{product}/candles"
GRANULARITY = 3600  # 1 hour
CHUNK = 300         # max candles per request


def fetch_range(product: str, start: datetime, end: datetime, session: requests.Session):
    """Yield [time, low, high, open, close, volume] rows for [start, end)."""
    cursor = start
    step = timedelta(seconds=GRANULARITY * CHUNK)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat(),
        }
        for attempt in range(5):
            resp = session.get(API.format(product=product), params=params, timeout=30)
            if resp.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"failed to fetch {cursor}: {resp.status_code} {resp.text[:200]}")
        rows = resp.json()
        yield from rows
        cursor = chunk_end
        time.sleep(0.15)  # stay well under the public rate limit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--out", default="data/btc_1h.csv")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    session = requests.Session()
    seen = {}
    total_chunks = 0
    for row in fetch_range(args.product, start, end, session):
        ts, low, high, open_, close, volume = row
        seen[int(ts)] = (int(ts), open_, high, low, close, volume)
        total_chunks += 1
        if total_chunks % 3000 == 0:
            print(f"  ... {len(seen)} candles so far", file=sys.stderr)

    ordered = [seen[k] for k in sorted(seen)]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        writer.writerows(ordered)

    first = datetime.fromtimestamp(ordered[0][0], tz=timezone.utc)
    last = datetime.fromtimestamp(ordered[-1][0], tz=timezone.utc)
    print(f"wrote {len(ordered)} candles to {args.out} ({first} .. {last})")


if __name__ == "__main__":
    main()
