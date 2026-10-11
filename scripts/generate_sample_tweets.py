"""Script to generate realistic synthetic sample tweets for offline replay."""

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "PFE", "WMT", "KO"]

BASE_TEMPLATES = [
    # Geopolitical & Macro
    (
        "${PREFIX} ${TICKER} facing supply chain friction due to new export controls and tariff threats in ${REGION}.",
        "Geopolitical",
    ),
    ("${PREFIX} Rumors spreading of sanctions hitting overseas subsidiaries of ${TICKER}.", "Geopolitical"),
    (
        "${PREFIX} Escalating regional military conflict in ${REGION} sparks energy supply concerns for ${TICKER}.",
        "Geopolitical",
    ),
    (
        "${PREFIX} Fed signals hawkish stance, higher rates putting pressure on bank margins for ${TICKER}.",
        "Monetary Policy",
    ),
    (
        "${PREFIX} Unexpected ${BPS} bps interest rate decision creates cross-asset volatility for ${TICKER}.",
        "Monetary Policy",
    ),
    (
        "${PREFIX} Persistent CPI inflation and rising yield curve inversion compressing forward guidance for ${TICKER}.",
        "Macroeconomic",
    ),
    ("${PREFIX} Macro headwinds and contraction in consumer spending weigh on ${TICKER} revenue.", "Macroeconomic"),
    # Credit & Default
    ("${PREFIX} Rating agency places ${TICKER} credit rating on negative watch citing high leverage.", "Credit Event"),
    (
        "${PREFIX} Spreads widening significantly by ${BPS} bps on ${TICKER} 5-year senior unsecured debt.",
        "Credit Event",
    ),
    (
        "${PREFIX} S&P and Moody's issue credit downgrade warning for ${TICKER} leveraged loan portfolio.",
        "Credit Event",
    ),
    ("${PREFIX} Rumors of covenant breach and debt restructuring surrounding ${TICKER} joint venture.", "Credit Event"),
    # Earnings & Corporate
    (
        "${PREFIX} ${TICKER} reports blowout Q${QTR} earnings, operating margins expand ${BPS} bps YoY!",
        "Earnings/Guidance",
    ),
    ("${PREFIX} ${TICKER} slashes FY forward revenue guidance citing slowing enterprise demand.", "Earnings/Guidance"),
    ("${PREFIX} Profit warning issued by ${TICKER} as operating costs surge in Q${QTR}.", "Earnings/Guidance"),
    (
        "${PREFIX} ${TICKER} announces emergency management restructuring and operational efficiency drive.",
        "Earnings/Guidance",
    ),
    # M&A & Antitrust
    (
        "${PREFIX} Regulators open antitrust investigation into ${TICKER}'s planned cloud acquisition.",
        "Regulatory/Legal",
    ),
    ("${PREFIX} DOJ prepares civil antitrust lawsuit against ${TICKER} over market dominance.", "Regulatory/Legal"),
    (
        "${PREFIX} ${TICKER} announces strategic buyout of key AI chip software vendor for $${AMOUNT}B.",
        "Merger/Acquisition",
    ),
    (
        "${PREFIX} Hostile takeover bid rumored for ${TICKER} regional competitor valued at $${AMOUNT}B.",
        "Merger/Acquisition",
    ),
    # Cyber & Disruption
    (
        "${PREFIX} Reports of major ransomware cyber attack affecting ${TICKER} internal enterprise databases.",
        "Cyber/Operational",
    ),
    ("${PREFIX} Severe cloud outage at ${TICKER} disrupts critical financial transaction feeds.", "Cyber/Operational"),
    # Supply & Commodities
    ("${PREFIX} Gulf coast hurricane forces ${TICKER} refinery shutdown, crude supplies tight.", "Natural Disaster"),
    (
        "${PREFIX} Key semiconductor substrate shortage halts manufacturing lines at ${TICKER}.",
        "Supply Chain/Commodity",
    ),
    (
        "${PREFIX} Port strikes and freight rate spikes create major logistics bottlenecks for ${TICKER}.",
        "Supply Chain/Commodity",
    ),
    # Product & General
    (
        "${PREFIX} ${TICKER} unveils next-generation enterprise AI platform with breakthrough performance.",
        "Product Launch",
    ),
    ("${PREFIX} FDA grants approval for ${TICKER}'s breakthrough medical therapy.", "Product Launch"),
    ("${PREFIX} Heavy institutional call option buying spotted on ${TICKER} ahead of investor symposium.", "Other"),
    ("${PREFIX} Wall Street analyst consensus downgrades ${TICKER} citing valuation multiples.", "Other"),
    (
        "${PREFIX} Strong operating cash flow generation and $${AMOUNT}B dividend increase announced by ${TICKER}.",
        "Earnings/Guidance",
    ),
    ("${PREFIX} Daily market wrap: trading volumes in ${TICKER} surge as sector rotates.", "Other"),
]

PREFIXES = [
    "Alert:",
    "Breaking:",
    "Market Watch:",
    "Flash:",
    "Newswire:",
    "Trading Desk:",
    "Headline:",
    "Update:",
    "Report:",
    "SCOOP:",
    "Intelligence:",
    "Terminal Alert:",
    "Flow:",
    "BofA Note:",
    "Desk Chat:",
]

REGIONS = ["Middle East", "Eastern Europe", "Asia-Pacific", "Latin America", "Gulf Region"]


def generate_sample_tweets(output_path: Path, n_rows: int = 360) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2026, 10, 8, 9, 0, 0, tzinfo=timezone.utc)

    rows = []
    seen_texts = set()
    attempts = 0

    while len(rows) < n_rows and attempts < n_rows * 10:
        attempts += 1
        ticker = random.choice(TICKERS)
        tpl, _ = random.choice(BASE_TEMPLATES)
        prefix = random.choice(PREFIXES)
        region = random.choice(REGIONS)
        bps = random.choice([25, 50, 75, 100, 150, 200])
        amount = random.choice([1.2, 2.5, 3.8, 5.0, 8.4, 12.5, 20.0])
        qtr = random.choice([1, 2, 3, 4])

        text = (
            tpl.replace("${PREFIX}", prefix)
            .replace("${TICKER}", ticker)
            .replace("${REGION}", region)
            .replace("${BPS}", str(bps))
            .replace("${AMOUNT}", str(amount))
            .replace("${QTR}", str(qtr))
        )

        if text in seen_texts:
            continue
        seen_texts.add(text)

        ts = base_time + timedelta(minutes=len(rows) * 11 + random.randint(1, 8))
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "ticker": ticker,
                "text": text,
                "user": f"trader_{random.randint(100, 999)}",
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "ticker", "text", "user"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {len(rows)} unique sample tweets at {output_path}")


if __name__ == "__main__":
    out = Path("data/sample_tweets.csv")
    generate_sample_tweets(out, 360)
