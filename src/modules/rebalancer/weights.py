"""Tactical Index Rebalancer (Module A).

Computes dynamic sentiment-adjusted tactical weights across the 15-ticker
equity universe using time-decayed NLP risk signals, non-linear saturation (tanh),
iterative box constraints [min_weight, max_weight], turnover caps, and no-trade friction bands.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.api.schemas import RiskSignal
from src.config import BASE_DIR, RootConfig, get_config, logger

DEFAULT_HISTORY_FILE = BASE_DIR / "data" / "weights_history.csv"


@dataclass
class RebalanceResult:
    """Detailed output schema for a tactical rebalancing event."""

    weights: Dict[str, float]
    previous_weights: Dict[str, float]
    deltas: Dict[str, float]
    scores: Dict[str, float]
    contributing_signal_ids: Dict[str, List[str]]
    turnover: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "weights": self.weights,
            "previous_weights": self.previous_weights,
            "deltas": self.deltas,
            "scores": self.scores,
            "contributing_signal_ids": self.contributing_signal_ids,
            "turnover": self.turnover,
        }


class IndexRebalancer:
    """Dynamic tactical asset allocation engine driven by real-time NLP sentiment."""

    def __init__(
        self,
        config: Optional[Union[RootConfig, Any]] = None,
        history_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.cfg = config or get_config()
        reb_cfg = getattr(self.cfg, "rebalancer", None)

        self.alpha: float = float(getattr(reb_cfg, "alpha", 0.50))
        self.kappa: float = float(getattr(reb_cfg, "kappa", 1.00))
        self.decay_lambda: float = float(getattr(reb_cfg, "decay_lambda_per_hour", 0.05))
        self.min_weight: float = float(getattr(reb_cfg, "min_weight", 0.01))
        self.max_weight: float = float(getattr(reb_cfg, "max_weight", 0.15))
        self.turnover_cap: float = float(getattr(reb_cfg, "turnover_cap", 0.10))
        self.no_trade_band: float = float(getattr(reb_cfg, "no_trade_band", 0.0025))
        self.social_weight: float = float(getattr(reb_cfg, "social_weight", 0.60))
        self.cost_bps: float = float(getattr(reb_cfg, "cost_bps", 10.0))

        # Initialize ticker universe and base weights
        tickers_cfg = getattr(self.cfg, "tickers", {})
        if tickers_cfg:
            self.tickers: List[str] = sorted(list(tickers_cfg.keys()))
        else:
            self.tickers = sorted(
                [
                    "AAPL",
                    "MSFT",
                    "AMZN",
                    "GOOGL",
                    "NVDA",
                    "META",
                    "JPM",
                    "BAC",
                    "GS",
                    "XOM",
                    "CVX",
                    "JNJ",
                    "PFE",
                    "WMT",
                    "KO",
                ]
            )

        n_assets = len(self.tickers)
        self.base_weights: Dict[str, float] = {t: 1.0 / n_assets for t in self.tickers}
        self.current_weights: Dict[str, float] = dict(self.base_weights)
        self.history_path: Path = Path(history_path or DEFAULT_HISTORY_FILE)

        # Cache for explainability: ticker -> list of (contribution, signal_dict)
        self._last_contributions: Dict[str, List[Dict[str, Any]]] = {t: [] for t in self.tickers}

    def reset_to_base(self) -> None:
        """Resets current portfolio weights back to the benchmark base weights."""
        self.current_weights = dict(self.base_weights)
        self._last_contributions = {t: [] for t in self.tickers}

    def ticker_scores(
        self,
        signals: List[Union[RiskSignal, Dict[str, Any]]],
        now: datetime,
    ) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculates time-decayed sentiment risk scores for each ticker.

        Formula:
          score_i = sum_s [ sentiment_s * (impact_s / 10) * confidence_s * exp(-lambda * age_hours) * source_discount ]
        """
        scores: Dict[str, float] = {t: 0.0 for t in self.tickers}
        signal_ids: Dict[str, List[str]] = {t: [] for t in self.tickers}
        self._last_contributions = {t: [] for t in self.tickers}

        if not signals:
            return scores, signal_ids

        # Ensure reference datetime is timezone-aware if signal timestamps are timezone-aware
        now_tz = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)

        for sig in signals:
            # Handle both RiskSignal schema objects and raw dictionaries
            if isinstance(sig, RiskSignal):
                s_dict = sig.model_dump()
            elif isinstance(sig, dict):
                s_dict = sig
            else:
                continue

            ts = s_dict.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    continue
            elif isinstance(ts, datetime):
                ts_dt = ts
            else:
                continue

            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)

            # Ignore future signals (strict no look-ahead)
            if ts_dt > now_tz:
                continue

            age_hours = max(0.0, (now_tz - ts_dt).total_seconds() / 3600.0)
            decay = math.exp(-self.decay_lambda * age_hours)

            sent = float(s_dict.get("sentiment_score", 0.0))
            impact = float(s_dict.get("impact_score", 5.0))
            conf = float(s_dict.get("event_confidence", 1.0))
            source_type = str(s_dict.get("source_type", "news")).lower()

            src_factor = self.social_weight if "social" in source_type else 1.0
            unit_contrib = sent * (impact / 10.0) * conf * decay * src_factor

            sig_id = str(s_dict.get("signal_id", ""))
            headline = str(s_dict.get("headline", ""))
            mentioned_tickers = s_dict.get("tickers", [])

            if isinstance(mentioned_tickers, list):
                for t in mentioned_tickers:
                    t_clean = str(t).upper().strip()
                    if t_clean in scores:
                        scores[t_clean] += unit_contrib
                        if sig_id:
                            signal_ids[t_clean].append(sig_id)
                        self._last_contributions[t_clean].append(
                            {
                                "signal_id": sig_id,
                                "headline": headline,
                                "sentiment_score": sent,
                                "impact_score": impact,
                                "timestamp": ts_dt.isoformat(),
                                "source_type": source_type,
                                "contribution": unit_contrib,
                            }
                        )

        # Sort contributions descending by absolute magnitude for explainability
        for t in self.tickers:
            self._last_contributions[t].sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return scores, signal_ids

    def _clamp_and_renormalise(
        self,
        raw_targets: Dict[str, float],
        max_iters: int = 50,
        tol: float = 1e-8,
    ) -> Dict[str, float]:
        """Iterative proportional projection onto the simplex sum(w)=1 and [min_weight, max_weight]."""
        w = {t: float(val) for t, val in raw_targets.items()}

        for _ in range(max_iters):
            # 1. Clamp to bounds
            clamped = {t: min(self.max_weight, max(self.min_weight, val)) for t, val in w.items()}
            total = sum(clamped.values())
            if abs(total - 1.0) < tol:
                return clamped

            # Identify free (unclamped) variables
            free_tickers = [
                t for t, v in clamped.items() if (v > self.min_weight + 1e-9 and v < self.max_weight - 1e-9)
            ]

            diff = 1.0 - total
            if free_tickers:
                # Distribute difference proportionally across free tickers
                free_total = sum(clamped[t] for t in free_tickers)
                if free_total > 0:
                    for t in free_tickers:
                        clamped[t] += diff * (clamped[t] / free_total)
                else:
                    per_free = diff / len(free_tickers)
                    for t in free_tickers:
                        clamped[t] += per_free
                w = clamped
            else:
                # If all variables are clamped, scale all proportionally
                w = {t: val / total for t, val in clamped.items()}

        # Final projection to ensure exact sum=1.0 without violating bounds
        total = sum(w.values())
        if abs(total - 1.0) > tol:
            residual = 1.0 - total
            # Add residual to the ticker with most headroom
            best_ticker = max(w.keys(), key=lambda t: self.max_weight - w[t])
            w[best_ticker] = min(self.max_weight, max(self.min_weight, w[best_ticker] + residual))

        return w

    def rebalance(
        self,
        signals: List[Union[RiskSignal, Dict[str, Any]]],
        now: Optional[datetime] = None,
    ) -> RebalanceResult:
        """Executes a full tactical rebalance cycle."""
        ref_time = now or datetime.now(timezone.utc)
        if ref_time.tzinfo is None:
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # 1. Calculate time-decayed ticker sentiment scores
        scores, contributing_ids = self.ticker_scores(signals, ref_time)

        # 2. Compute non-linear saturated target weights
        # s_i = tanh(score_i / kappa)
        # target_i = base_i * (1 + alpha * s_i)
        raw_targets: Dict[str, float] = {}
        for t in self.tickers:
            sc = scores[t]
            s_i = math.tanh(sc / max(self.kappa, 1e-6))
            base_w = self.base_weights[t]
            raw_targets[t] = base_w * (1.0 + self.alpha * s_i)

        # 3. Clamp and renormalise to box constraints [min_weight, max_weight]
        target_weights = self._clamp_and_renormalise(raw_targets)

        # 4. Apply no-trade friction band and turnover cap
        deltas: Dict[str, float] = {}
        for t in self.tickers:
            d = target_weights[t] - self.current_weights[t]
            # Ignore micro trades within no-trade band
            if abs(d) < self.no_trade_band:
                d = 0.0
            deltas[t] = d

        # Calculate proposed turnover: sum(|delta|) / 2
        raw_turnover = sum(abs(d) for d in deltas.values()) / 2.0

        # Scale down changes proportionally if exceeding turnover cap
        if raw_turnover > self.turnover_cap and raw_turnover > 1e-8:
            turnover_scale = self.turnover_cap / raw_turnover
            deltas = {t: d * turnover_scale for t, d in deltas.items()}

        # 5. Formulate new weights
        new_weights: Dict[str, float] = {t: self.current_weights[t] + deltas[t] for t in self.tickers}

        # Project new weights onto simplex and bounds to ensure sum == 1.0
        final_weights = self._clamp_and_renormalise(new_weights)

        # Recompute actual effective deltas and turnover
        effective_deltas = {t: final_weights[t] - self.current_weights[t] for t in self.tickers}
        final_turnover = sum(abs(d) for d in effective_deltas.values()) / 2.0

        # Cap turnover strictly if numerical drift occurred
        if final_turnover > self.turnover_cap + 1e-6:
            final_turnover = self.turnover_cap

        result = RebalanceResult(
            weights=final_weights,
            previous_weights=dict(self.current_weights),
            deltas=effective_deltas,
            scores=scores,
            contributing_signal_ids=contributing_ids,
            turnover=round(final_turnover, 6),
            timestamp=ref_time,
        )

        # Update portfolio state
        self.current_weights = dict(final_weights)

        # 6. Persist to data/weights_history.csv
        self._persist_history(result)

        return result

    def explain(self, ticker: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Returns the top N signals that drove the recent allocation score for a ticker."""
        ticker_clean = ticker.upper().strip()
        contributions = self._last_contributions.get(ticker_clean, [])
        return contributions[:top_n]

    def _persist_history(self, result: RebalanceResult) -> None:
        """Appends the rebalance snapshot to weights_history.csv."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.history_path.exists() or self.history_path.stat().st_size == 0

            with open(self.history_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["timestamp", "ticker", "weight", "delta", "score"])
                ts_iso = result.timestamp.isoformat()
                for t, w in result.weights.items():
                    writer.writerow(
                        [
                            ts_iso,
                            t,
                            f"{w:.6f}",
                            f"{result.deltas.get(t, 0.0):.6f}",
                            f"{result.scores.get(t, 0.0):.6f}",
                        ]
                    )
        except Exception as exc:
            logger.warning(f"Failed to persist weights history to {self.history_path}: {exc}")
