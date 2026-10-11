"""Tactical Rebalancing Backtest Engine (Module A).

Performs strict, point-in-time, no-look-ahead historical backtesting:
- Ingests prices for the 15-ticker index universe (yfinance with local CSV fallback).
- Rebalances at each trading boundary using ONLY signals with timestamp <= t.
- Benchmarks against an equal-weight buy-and-hold index.
- Computes cumulative returns, annualised volatility, Sharpe ratio, max drawdown,
  turnover, and transaction-cost-adjusted net returns.
- Persists backtest metrics to docs/results/backtest.json and equity curve to data/backtest_curve.csv.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf

from src.api.schemas import RiskSignal
from src.config import BASE_DIR, RootConfig, get_config, logger
from src.modules.rebalancer.weights import IndexRebalancer
from src.storage.store import get_signal_store

PRICES_CACHE_PATH = BASE_DIR / "data" / "raw" / "prices_cache.csv"
TWEETS_PATH = BASE_DIR / "data" / "sample_tweets.csv"
RESULTS_JSON_PATH = BASE_DIR / "docs" / "results" / "backtest.json"
EQUITY_CURVE_PATH = BASE_DIR / "data" / "backtest_curve.csv"


def get_tweet_date_range(tweets_file: Optional[Path] = None) -> Tuple[datetime, datetime]:
    """Inspects the tweet dataset to determine the evaluation time window."""
    file_path = tweets_file or TWEETS_PATH
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            if "timestamp" in df.columns:
                ts_series = pd.to_datetime(df["timestamp"])
                start_dt = ts_series.min().to_pydatetime()
                end_dt = ts_series.max().to_pydatetime()
                # Ensure timezone-aware UTC
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                return start_dt, end_dt
        except Exception as exc:
            logger.warning(f"Could not parse tweet timestamps from {file_path}: {exc}")

    # Fallback: past 12 months
    end_now = datetime.now(timezone.utc)
    start_fallback = end_now - timedelta(days=365)
    return start_fallback, end_now


def load_or_fetch_prices(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    cache_path: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Retrieves adjusted close prices using yfinance with resilient local CSV fallback."""
    c_path = cache_path or PRICES_CACHE_PATH

    # 1. Try downloading if requested or if cache does not exist
    if force_download or not c_path.exists():
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            # Pad end date by 1 day for inclusive range
            end_str = (end_date + timedelta(days=2)).strftime("%Y-%m-%d")
            logger.info(f"Downloading historical price data for {len(tickers)} tickers ({start_str} to {end_str})...")

            raw = yf.download(
                tickers=tickers,
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True,
            )

            if isinstance(raw, pd.DataFrame) and not raw.empty:
                prices_df = raw["Close"] if "Close" in raw else raw
                if all(t in prices_df.columns for t in tickers):
                    prices_df = prices_df[tickers].dropna()
                    if not prices_df.empty:
                        c_path.parent.mkdir(parents=True, exist_ok=True)
                        prices_df.to_csv(c_path)
                        logger.info(f"Saved freshly fetched prices to {c_path}")
                        return prices_df
        except Exception as exc:
            logger.warning(f"yfinance download failed: {exc}. Attempting local cache fallback.")

    # 2. Local cache fallback
    if c_path.exists():
        try:
            df = pd.read_csv(c_path, index_col=0, parse_dates=True)
            # Filter for requested tickers that exist in the CSV
            valid_tickers = [t for t in tickers if t in df.columns]
            if len(valid_tickers) == len(tickers):
                df = df[valid_tickers].ffill().bfill()
                logger.info(f"Loaded {len(df)} price observations from local cache: {c_path}")
                return df
        except Exception as exc:
            logger.error(f"Failed to read prices cache at {c_path}: {exc}")

    raise RuntimeError(f"Could not obtain price data for tickers {tickers} online or offline.")


def calculate_metrics(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Calculates cumulative return, annualized volatility, Sharpe, max drawdown, and turnover."""
    if len(returns) == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "average_turnover": 0.0,
        }

    # Cumulative return: prod(1 + r) - 1
    wealth_index = (1.0 + returns).cumprod()
    cum_ret = float(wealth_index.iloc[-1] - 1.0) if len(wealth_index) > 0 else 0.0

    # Annualised volatility: std(r) * sqrt(252)
    daily_vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    ann_vol = daily_vol * np.sqrt(252.0)

    # Sharpe ratio (risk-free 0.0)
    daily_mean = float(returns.mean())
    sharpe = (daily_mean * 252.0 / ann_vol) if ann_vol > 1e-8 else 0.0

    # Max Drawdown
    running_max = wealth_index.cummax()
    drawdown = (wealth_index - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    avg_turnover = float(turnover.mean()) if turnover is not None and len(turnover) > 0 else 0.0

    return {
        "cumulative_return": round(cum_ret, 6),
        "annualized_volatility": round(ann_vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "average_turnover": round(avg_turnover, 6),
    }


def run_backtest(
    prices_df: pd.DataFrame,
    signals: List[Union[RiskSignal, Dict[str, Any]]],
    config: Optional[Union[RootConfig, Any]] = None,
    cost_bps: Optional[float] = None,
) -> Dict[str, Any]:
    """Executes a strict point-in-time tactical rebalancing simulation.

    At each trading date t:
      - ONLY signals with timestamp <= t are accessible (zero look-ahead bias).
      - Rebalance weights are generated and applied to the forward return from t to t+1.
      - Transaction costs = turnover * (cost_bps / 10,000).
    """
    cfg = config or get_config()
    rebalancer = IndexRebalancer(config=cfg)
    tickers = rebalancer.tickers

    if cost_bps is None:
        cost_bps = rebalancer.cost_bps

    # Validate price dataframe
    missing_cols = [t for t in tickers if t not in prices_df.columns]
    if missing_cols:
        raise ValueError(f"Prices DataFrame missing required ticker columns: {missing_cols}")

    trading_dates = prices_df.index.tolist()
    if len(trading_dates) < 2:
        raise ValueError("Need at least 2 price dates to run backtest.")

    records: List[Dict[str, Any]] = []

    # Initialize equity indices at 1.0
    strategy_equity = 1.0
    benchmark_equity = 1.0
    net_strategy_equity = 1.0

    rebalancer.reset_to_base()

    for i in range(len(trading_dates) - 1):
        t_current = trading_dates[i]
        t_next = trading_dates[i + 1]

        # Convert t_current to timezone-aware UTC datetime for strict point-in-time filtering
        if isinstance(t_current, pd.Timestamp):
            t_dt = t_current.to_pydatetime()
        else:
            t_dt = t_current

        if t_dt.tzinfo is None:
            t_dt = t_dt.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        else:
            t_dt = t_dt.replace(hour=23, minute=59, second=59)

        # 1. Strict point-in-time signal slice (NO LOOK-AHEAD)
        available_signals = []
        for s in signals:
            s_ts = s.timestamp if isinstance(s, RiskSignal) else s.get("timestamp")
            if isinstance(s_ts, str):
                s_dt = datetime.fromisoformat(s_ts.replace("Z", "+00:00"))
            elif isinstance(s_ts, datetime):
                s_dt = s_ts
            else:
                continue

            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=timezone.utc)

            if s_dt <= t_dt:
                available_signals.append(s)

        # 2. Rebalance weights
        rebal_res = rebalancer.rebalance(available_signals, now=t_dt)
        strat_weights = rebal_res.weights
        turnover = rebal_res.turnover

        # 3. Calculate forward returns from t_current to t_next
        p_now = prices_df.loc[t_current, tickers].values.astype(float)
        p_next = prices_df.loc[t_next, tickers].values.astype(float)

        # Safe single-period asset returns
        asset_returns = np.where(p_now > 1e-8, (p_next - p_now) / p_now, 0.0)

        # Strategy return (tactical weights)
        w_strat_vec = np.array([strat_weights[t] for t in tickers])
        strat_ret = float(np.dot(w_strat_vec, asset_returns))

        # Benchmark return (equal-weight benchmark)
        w_bench_vec = np.array([rebalancer.base_weights[t] for t in tickers])
        bench_ret = float(np.dot(w_bench_vec, asset_returns))

        # Transaction cost impact
        tc = turnover * (cost_bps / 10000.0)
        net_strat_ret = strat_ret - tc

        # Update wealth indices
        strategy_equity *= 1.0 + strat_ret
        benchmark_equity *= 1.0 + bench_ret
        net_strategy_equity *= 1.0 + net_strat_ret

        records.append(
            {
                "date": t_next.strftime("%Y-%m-%d") if isinstance(t_next, pd.Timestamp) else str(t_next),
                "strategy_equity": round(strategy_equity, 6),
                "benchmark_equity": round(benchmark_equity, 6),
                "net_strategy_equity": round(net_strategy_equity, 6),
                "strategy_return": round(strat_ret, 6),
                "benchmark_return": round(bench_ret, 6),
                "net_return": round(net_strat_ret, 6),
                "turnover": round(turnover, 6),
                "active_signals_count": len(available_signals),
            }
        )

    curve_df = pd.DataFrame(records)

    # Performance Analytics
    strat_metrics = calculate_metrics(curve_df["strategy_return"], curve_df["turnover"])
    bench_metrics = calculate_metrics(curve_df["benchmark_return"], None)
    net_metrics = calculate_metrics(curve_df["net_return"], curve_df["turnover"])

    # Alpha / Excess return
    excess_ret = strat_metrics["cumulative_return"] - bench_metrics["cumulative_return"]
    net_excess_ret = net_metrics["cumulative_return"] - bench_metrics["cumulative_return"]

    summary = {
        "evaluation_period": {
            "start_date": curve_df["date"].iloc[0] if not curve_df.empty else "",
            "end_date": curve_df["date"].iloc[-1] if not curve_df.empty else "",
            "trading_days": len(curve_df),
        },
        "transaction_cost_bps": cost_bps,
        "strategy": strat_metrics,
        "strategy_net_of_cost": net_metrics,
        "benchmark_equal_weight": bench_metrics,
        "excess_cumulative_return": round(excess_ret, 6),
        "net_excess_cumulative_return": round(net_excess_ret, 6),
        "limitations_note": (
            "The evaluation sample period is short, and historical backtest results "
            "are illustrative and intended solely for model validation. Past performance does "
            "not guarantee future results, and no claim of trading profitability is made."
        ),
    }

    # Persist artifacts
    try:
        EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        curve_df.to_csv(EQUITY_CURVE_PATH, index=False)
        logger.info(f"Saved backtest equity curve to {EQUITY_CURVE_PATH}")

        RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved backtest metrics to {RESULTS_JSON_PATH}")
    except Exception as exc:
        logger.warning(f"Failed to write backtest result files: {exc}")

    return summary


def main() -> None:
    """CLI Entry point to execute the rebalancer backtest."""
    print("=" * 75)
    print("RiskPulse Module A: Tactical Rebalancing Engine Backtest")
    print("=" * 75)

    cfg = get_config()
    store = get_signal_store()
    signals = store.query(limit=1000)
    print(f"Loaded {len(signals)} historical signals from database.")

    # Determine date range from tweet dataset
    start_dt, end_dt = get_tweet_date_range()
    print(f"Dataset date window: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

    # Universe tickers
    rebalancer = IndexRebalancer(config=cfg)
    tickers = rebalancer.tickers

    # Obtain price history
    prices_df = load_or_fetch_prices(tickers, start_dt, end_dt)

    # Run backtest
    results = run_backtest(prices_df=prices_df, signals=signals, config=cfg)

    print("\n--- Backtest Performance Results ---")
    print(f"Trading Days:       {results['evaluation_period']['trading_days']}")
    print(f"Benchmark Return:   {results['benchmark_equal_weight']['cumulative_return']:+.2%}")
    print(f"Benchmark Sharpe:   {results['benchmark_equal_weight']['sharpe_ratio']:.2f}")
    print(f"Strategy Return:    {results['strategy']['cumulative_return']:+.2%}")
    print(f"Strategy Sharpe:    {results['strategy']['sharpe_ratio']:.2f}")
    print(f"Net (Cost-Adj) Ret: {results['strategy_net_of_cost']['cumulative_return']:+.2%}")
    print(f"Net Sharpe:         {results['strategy_net_of_cost']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {results['strategy_net_of_cost']['max_drawdown']:.2%}")
    print(f"Average Turnover:   {results['strategy_net_of_cost']['average_turnover']:.2%}")

    print("\n" + "*" * 75)
    print("HONEST LIMITATIONS NOTE:")
    print(results["limitations_note"])
    print("*" * 75 + "\n")


if __name__ == "__main__":
    main()
