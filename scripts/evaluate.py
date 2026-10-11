"""Empirical evaluation suite for RiskPulse AI/NLP Risk Engine.

Measures:
  A) Financial Sentiment: Naive Lexicon vs VADER vs FinBERT (Accuracy, Macro-F1, Confusion Matrix)
  B) Event Classification: Rules-only vs Hybrid (Zero-shot + Rules) on labelled events
  C) Market Impact & Sentiment Validation: Spearman correlation against yfinance market returns
  D) Processing Speed & Throughput: Documents/sec on CPU
  E) Exports: metrics.json, metrics.md, and visualization PNGs in docs/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Computes subset accuracy."""
    return sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true) if y_true else 0.0


def confusion_matrix(y_true: List[Any], y_pred: List[Any], labels: List[Any]) -> np.ndarray:
    """Computes confusion matrix matching scikit-learn convention."""
    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            mat[idx[yt], idx[yp]] += 1
    return mat


def precision_recall_fscore_support(
    y_true: List[Any],
    y_pred: List[Any],
    labels: List[Any],
    zero_division: float = 0.0,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Computes precision, recall, F1, and support per class."""
    precisions: List[float] = []
    recalls: List[float] = []
    fscores: List[float] = []
    supports: List[int] = []
    for c in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        p = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        r = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f = (2 * p * r) / (p + r) if (p + r) > 0 else zero_division
        s = sum(1 for yt in y_true if yt == c)
        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
        supports.append(s)
    return precisions, recalls, fscores, supports


def f1_score(
    y_true: List[Any],
    y_pred: List[Any],
    labels: List[Any],
    average: str = "macro",
    zero_division: float = 0.0,
) -> float:
    """Computes macro F1-score across specified labels."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=zero_division)
    return sum(f) / len(f) if f else 0.0


sys.path.insert(0, str(BASE_DIR))

from src.config import logger
from src.ingestion.normalise import NormalizedDoc
from src.nlp.engine import RiskEngine
from src.nlp.events import EventClassifier
from src.nlp.sentiment import SentimentAnalyzer

RESULTS_DIR = BASE_DIR / "docs" / "results"
RAW_DIR = BASE_DIR / "data" / "raw"


# ---------------------------------------------------------------------------
# A) SENTIMENT BENCHMARK
# ---------------------------------------------------------------------------


class NaiveFinancialLexicon:
    """Simple baseline lexicon for financial sentiment classification."""

    POSITIVE_WORDS = {
        "profit",
        "profits",
        "profitable",
        "growth",
        "growing",
        "grew",
        "gains",
        "gain",
        "increase",
        "increased",
        "increasing",
        "surge",
        "surged",
        "surging",
        "jump",
        "jumped",
        "beat",
        "beats",
        "beating",
        "rise",
        "rose",
        "rising",
        "record",
        "high",
        "higher",
        "highest",
        "positive",
        "exceed",
        "exceeded",
        "outperform",
        "dividend",
        "strong",
        "stronger",
        "boost",
        "boosted",
        "rally",
        "rallied",
        "up",
    }

    NEGATIVE_WORDS = {
        "loss",
        "losses",
        "fall",
        "fell",
        "falling",
        "drop",
        "dropped",
        "dropping",
        "decline",
        "declined",
        "declining",
        "decrease",
        "decreased",
        "slump",
        "slumped",
        "plunge",
        "plunged",
        "plunging",
        "cut",
        "cuts",
        "cutting",
        "down",
        "warning",
        "warned",
        "warns",
        "low",
        "lower",
        "lowest",
        "negative",
        "weak",
        "weaker",
        "weakness",
        "miss",
        "missed",
        "missing",
        "deficit",
        "inflation",
        "recession",
        "default",
        "bankruptcy",
        "downgrade",
        "downgraded",
    }

    def predict(self, text: str) -> str:
        words = set(text.lower().split())
        pos_hits = len(words & self.POSITIVE_WORDS)
        neg_hits = len(words & self.NEGATIVE_WORDS)
        if pos_hits > neg_hits:
            return "positive"
        elif neg_hits > pos_hits:
            return "negative"
        return "neutral"


def evaluate_sentiment(sample_size: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Evaluates Naive Lexicon, VADER, and FinBERT on Kaggle Financial PhraseBank."""
    logger.info(f"--- Running Sentiment Evaluation (N={sample_size}, Seed={seed}) ---")

    # Locate CSV in data/raw/
    csv_candidates = [
        RAW_DIR / "sentiment_analysis_for_financial_news.csv",
        RAW_DIR / "all-data.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.is_file()), None)
    if not csv_path:
        raise FileNotFoundError(f"Financial sentiment CSV not found in {RAW_DIR}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "sentiment" not in df.columns or "text" not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'sentiment' and 'text' columns")

    # Clean and filter labels
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    valid_mask = df["sentiment"].isin(["positive", "neutral", "negative"]) & df["text"].notna()
    df_valid = df[valid_mask].copy()

    # Sample evaluation split
    sample_n = min(sample_size, len(df_valid))
    eval_df = df_valid.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    y_true = eval_df["sentiment"].tolist()
    texts = eval_df["text"].tolist()

    labels_order = ["negative", "neutral", "positive"]

    # 1. Naive Keyword Baseline
    logger.info("Evaluating Naive Keyword baseline...")
    naive = NaiveFinancialLexicon()
    y_pred_naive = [naive.predict(t) for t in texts]

    # 2. VADER
    logger.info("Evaluating VADER financial analyzer...")
    vader_analyzer = SentimentAnalyzer(backend_override="vader")
    vader_results = vader_analyzer.analyze(texts)
    y_pred_vader = [r.label for r in vader_results]

    # 3. FinBERT
    logger.info("Evaluating FinBERT neural model...")
    finbert_analyzer = SentimentAnalyzer(backend_override="finbert")
    finbert_results = finbert_analyzer.analyze(texts)
    y_pred_finbert = [r.label for r in finbert_results]

    # Metrics computation
    def compute_metrics(y_pred: List[str]) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels_order, average="macro"))
        cm = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()
        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": cm,
            "labels": labels_order,
        }

    metrics_naive = compute_metrics(y_pred_naive)
    metrics_vader = compute_metrics(y_pred_vader)
    metrics_finbert = compute_metrics(y_pred_finbert)

    # Generate Sentiment Comparison Bar Chart
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    methods = ["Naive Lexicon", "Enhanced VADER", "FinBERT"]
    accs = [metrics_naive["accuracy"], metrics_vader["accuracy"], metrics_finbert["accuracy"]]
    f1s = [metrics_naive["macro_f1"], metrics_vader["macro_f1"], metrics_finbert["macro_f1"]]

    x = np.arange(len(methods))
    width = 0.35
    rects1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#1e88e5")
    rects2 = ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="#43a047")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title(
        "Financial Sentiment Model Performance Comparison (Kaggle FinancialPhraseBank)", fontsize=11, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    comp_path = RESULTS_DIR / "sentiment_comparison.png"
    plt.savefig(comp_path, dpi=200)
    plt.close()
    logger.info(f"Saved sentiment comparison chart: {comp_path}")

    # Generate FinBERT Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    cm_arr = np.array(metrics_finbert["confusion_matrix"])
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=labels_order,
        yticklabels=labels_order,
        title="FinBERT Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
                fontweight="bold",
            )
    plt.tight_layout()
    cm_path = RESULTS_DIR / "finbert_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Saved FinBERT confusion matrix: {cm_path}")

    return {
        "sample_size": sample_n,
        "naive_lexicon": metrics_naive,
        "vader": metrics_vader,
        "finbert": metrics_finbert,
    }


# ---------------------------------------------------------------------------
# B) EVENTS CLASSIFICATION BENCHMARK
# ---------------------------------------------------------------------------


def evaluate_events() -> Dict[str, Any]:
    """Evaluates rules-only vs hybrid zero-shot event classification."""
    logger.info("--- Running Event Classification Evaluation ---")
    events_csv = BASE_DIR / "data" / "labelled_events.csv"
    if not events_csv.is_file():
        raise FileNotFoundError(f"Labelled events dataset not found: {events_csv}")

    df = pd.read_csv(events_csv)
    y_true = df["label"].tolist()
    texts = df["text"].tolist()
    classes = sorted(list(set(y_true)))

    # 1. Rules-only
    logger.info("Evaluating rules-only classifier...")
    clf_rules = EventClassifier(use_zero_shot=False)
    y_pred_rules = [clf_rules.classify(t).event_type for t in texts]

    # 2. Hybrid (Zero-shot + Rules)
    logger.info("Evaluating hybrid (zero-shot + rules) classifier...")
    clf_hybrid = EventClassifier(use_zero_shot=True)
    y_pred_hybrid = [clf_hybrid.classify(t).event_type for t in texts]

    def build_eval_report(y_pred: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
        per_class = {}
        for cls_name, prec, rec, fscore, supp in zip(classes, p, r, f, s):
            per_class[cls_name] = {
                "precision": round(float(prec), 3),
                "recall": round(float(rec), 3),
                "f1": round(float(fscore), 3),
                "support": int(supp),
            }
        macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

        errors = []
        for t, yt, yp in zip(texts, y_true, y_pred):
            if yt != yp:
                errors.append({"text": t, "true_label": yt, "predicted_label": yp})

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class,
            "confusion_matrix": cm,
            "classes": classes,
        }, errors

    rep_rules, err_rules = build_eval_report(y_pred_rules)
    rep_hybrid, err_hybrid = build_eval_report(y_pred_hybrid)

    # Save Event Confusion Matrix Heatmap (Hybrid)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cm_arr = np.array(rep_hybrid["confusion_matrix"])
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title="Hybrid Event Classifier Confusion Matrix",
        ylabel="True Category",
        xlabel="Predicted Category",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = cm_arr[i, j]
            if val > 0:
                ax.text(
                    j,
                    i,
                    format(val, "d"),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8,
                    fontweight="bold",
                )
    plt.tight_layout()
    cm_path = RESULTS_DIR / "events_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    logger.info(f"Saved event confusion matrix: {cm_path}")

    return {
        "dataset_size": len(df),
        "rules_only": rep_rules,
        "hybrid": rep_hybrid,
        "common_errors": err_hybrid[:10],
    }


# ---------------------------------------------------------------------------
# C) IMPACT & SENTIMENT MARKET VALIDATION
# ---------------------------------------------------------------------------


def evaluate_impact_validation() -> Dict[str, Any]:
    """Validates impact scores and sentiment scores against historical stock price movements."""
    logger.info("--- Running Impact & Sentiment Market Validation (yfinance) ---")
    signals_file = BASE_DIR / "data" / "signals_sample.jsonl"
    if not signals_file.is_file():
        raise FileNotFoundError(f"Signals file not found: {signals_file}")

    with open(signals_file, "r", encoding="utf-8") as f:
        signals = [json.loads(line) for line in f]

    # Collect tickers
    all_tickers = sorted(list({t for s in signals for t in s.get("tickers", [])}))
    if not all_tickers:
        raise ValueError("No tickers found in signals sample")

    # Load or download price cache
    cache_file = RAW_DIR / "prices_cache.csv"
    if cache_file.is_file():
        logger.info(f"Loading cached prices from {cache_file}...")
        prices_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        logger.info(f"Downloading prices for {len(all_tickers)} tickers via yfinance...")
        import yfinance as yf

        prices_raw = yf.download(all_tickers, period="6mo", progress=False)
        prices_df = prices_raw["Close"] if "Close" in prices_raw else prices_raw
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices_df.to_csv(cache_file)
        logger.info(f"Cached prices to {cache_file}")

    trading_days = pd.DatetimeIndex(prices_df.index).tz_localize(None).normalize()

    valid_samples = []
    for s in signals:
        tickers = s.get("tickers", [])
        if not tickers:
            continue
        ticker = tickers[0]
        if ticker not in prices_df.columns:
            continue

        ts = pd.to_datetime(s["timestamp"]).tz_localize(None).normalize()
        prior_days = trading_days[trading_days <= ts]
        if len(prior_days) == 0:
            continue
        t0 = prior_days[-1]

        future_days = trading_days[trading_days > t0]
        if len(future_days) < 1:
            continue
        t1 = future_days[0]

        p0 = prices_df.loc[prices_df.index.normalize() == t0, ticker].values[0]
        p1 = prices_df.loc[prices_df.index.normalize() == t1, ticker].values[0]

        if np.isnan(p0) or np.isnan(p1) or p0 <= 0:
            continue

        ret_1d = float((p1 - p0) / p0)
        abs_ret_1d = float(abs(ret_1d))

        if len(future_days) >= 3:
            t2 = future_days[1]
            t3 = future_days[2]
            p2 = prices_df.loc[prices_df.index.normalize() == t2, ticker].values[0]
            p3 = prices_df.loc[prices_df.index.normalize() == t3, ticker].values[0]
            r1 = (p1 - p0) / p0
            r2 = (p2 - p1) / p1
            r3 = (p3 - p2) / p2
            vol_3d = float(np.std([r1, r2, r3]))
        else:
            vol_3d = abs_ret_1d

        valid_samples.append(
            {
                "signal_id": s["signal_id"],
                "ticker": ticker,
                "headline": s.get("headline", ""),
                "impact_score": float(s["impact_score"]),
                "sentiment_score": float(s["sentiment_score"]),
                "ret_1d": ret_1d,
                "abs_ret_1d": abs_ret_1d,
                "vol_3d": vol_3d,
            }
        )

    logger.info(f"Aligned {len(valid_samples)} signals with historical market data.")
    if len(valid_samples) < 10:
        raise ValueError(f"Insufficient aligned signal-price samples ({len(valid_samples)})")

    df_samples = pd.DataFrame(valid_samples)

    # Statistical Correlations (Spearman Rank)
    sp_impact = spearmanr(df_samples["impact_score"], df_samples["abs_ret_1d"])
    sp_vol = spearmanr(df_samples["impact_score"], df_samples["vol_3d"])
    sp_sent = spearmanr(df_samples["sentiment_score"], df_samples["ret_1d"])

    # Generate Dual Scatter Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    # Subplot 1: Impact vs Abs Next-Day Return
    ax1.scatter(
        df_samples["impact_score"], df_samples["abs_ret_1d"] * 100, color="#d32f2f", alpha=0.6, edgecolors="none", s=40
    )
    z1 = np.polyfit(df_samples["impact_score"], df_samples["abs_ret_1d"] * 100, 1)
    p1 = np.poly1d(z1)
    x_vals1 = np.linspace(df_samples["impact_score"].min(), df_samples["impact_score"].max(), 50)
    ax1.plot(x_vals1, p1(x_vals1), color="#b71c1c", linestyle="--", linewidth=1.5, label="Linear Trend")
    ax1.set_xlabel("Predicted Impact Score [1, 10]", fontweight="bold")
    ax1.set_ylabel("Next-Trading-Day |Return| (%)", fontweight="bold")
    ax1.set_title(
        f"Impact Severity vs Market Move\nSpearman ρ = {sp_impact.statistic:.3f} (p = {sp_impact.pvalue:.4f})",
        fontsize=10,
    )
    ax1.legend(loc="upper left")

    # Subplot 2: Sentiment vs Signed Next-Day Return
    ax2.scatter(
        df_samples["sentiment_score"], df_samples["ret_1d"] * 100, color="#1976d2", alpha=0.6, edgecolors="none", s=40
    )
    z2 = np.polyfit(df_samples["sentiment_score"], df_samples["ret_1d"] * 100, 1)
    p2 = np.poly1d(z2)
    x_vals2 = np.linspace(df_samples["sentiment_score"].min(), df_samples["sentiment_score"].max(), 50)
    ax2.plot(x_vals2, p2(x_vals2), color="#0d47a1", linestyle="--", linewidth=1.5, label="Linear Trend")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Sentiment Score [-1.0, 1.0]", fontweight="bold")
    ax2.set_ylabel("Next-Trading-Day Return (%)", fontweight="bold")
    ax2.set_title(
        f"Sentiment vs Signed Market Return\nSpearman ρ = {sp_sent.statistic:.3f} (p = {sp_sent.pvalue:.4f})",
        fontsize=10,
    )
    ax2.legend(loc="upper left")

    plt.suptitle("Market Impact Validation (yfinance Daily Historical Prices)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    scatter_path = RESULTS_DIR / "impact_validation_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    logger.info(f"Saved impact validation scatter plot: {scatter_path}")

    return {
        "sample_size": len(valid_samples),
        "impact_vs_abs_return": {
            "spearman_rho": round(float(sp_impact.statistic), 4),
            "p_value": round(float(sp_impact.pvalue), 4),
        },
        "impact_vs_volatility": {
            "spearman_rho": round(float(sp_vol.statistic), 4),
            "p_value": round(float(sp_vol.pvalue), 4),
        },
        "sentiment_vs_return": {
            "spearman_rho": round(float(sp_sent.statistic), 4),
            "p_value": round(float(sp_sent.pvalue), 4),
        },
        "disclaimer": "Empirical correlation reflects short-term market co-movement across historical observation dates and does NOT imply direct causal determinism.",
    }


# ---------------------------------------------------------------------------
# D) SPEED & THROUGHPUT BENCHMARK
# ---------------------------------------------------------------------------


def evaluate_speed(n_docs: int = 200) -> Dict[str, Any]:
    """Measures documents processed per second on CPU across backends."""
    logger.info(f"--- Running Speed & Throughput Benchmark ({n_docs} documents, CPU only) ---")
    tweets_csv = BASE_DIR / "data" / "sample_tweets.csv"
    if not tweets_csv.is_file():
        raise FileNotFoundError(f"Tweets CSV not found: {tweets_csv}")

    df_tweets = pd.read_csv(tweets_csv)
    raw_texts = df_tweets["text"].dropna().tolist()
    if len(raw_texts) < n_docs:
        raw_texts = (raw_texts * (n_docs // len(raw_texts) + 1))[:n_docs]
    else:
        raw_texts = raw_texts[:n_docs]

    test_docs = [
        NormalizedDoc.create(
            source="tweets",
            source_type="social",
            published_at=datetime.now(timezone.utc),
            title=t[:60],
            text=t,
        )
        for t in raw_texts
    ]

    # Benchmark VADER + Rules pipeline (Fast production / offline mode)
    engine_vader = RiskEngine(sentiment_backend="vader", use_zero_shot=False)
    t0 = time.perf_counter()
    res_vader = engine_vader.process(test_docs)
    elapsed_vader = time.perf_counter() - t0
    rate_vader = len(res_vader) / max(elapsed_vader, 1e-6)

    # Benchmark FinBERT + Rules pipeline
    engine_finbert = RiskEngine(sentiment_backend="finbert", use_zero_shot=False)
    t0_fb = time.perf_counter()
    res_finbert = engine_finbert.process(test_docs[:50])  # 50 doc probe on CPU
    elapsed_fb = time.perf_counter() - t0_fb
    rate_finbert = len(res_finbert) / max(elapsed_fb, 1e-6)

    return {
        "n_docs": n_docs,
        "vader_rules": {
            "processed": len(res_vader),
            "duration_seconds": round(elapsed_vader, 3),
            "docs_per_second": round(rate_vader, 2),
        },
        "finbert_rules": {
            "processed": len(res_finbert),
            "duration_seconds": round(elapsed_fb, 3),
            "docs_per_second": round(rate_finbert, 2),
        },
    }


# ---------------------------------------------------------------------------
# E) MARKDOWN & JSON EXPORT
# ---------------------------------------------------------------------------


def generate_markdown_report(metrics: Dict[str, Any]) -> str:
    """Generates comprehensive markdown report with clean tables."""
    sent = metrics["sentiment"]
    ev = metrics["events"]
    imp = metrics["impact_validation"]
    spd = metrics["speed"]

    md = []
    md.append("# RiskPulse AI/NLP Engine Empirical Evaluation Report")
    md.append(
        f"\n*Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} | Platform: CPU Benchmark*"
    )
    md.append("\n---\n")

    md.append("## 1. Financial Sentiment Benchmark")
    md.append(
        f"Evaluated on a random sample of **{sent['sample_size']} financial headlines** from the Kaggle FinancialPhraseBank dataset (`seed=42`).\n"
    )
    md.append("| Model / Approach | Accuracy | Macro-F1 | Precision/Recall Notes |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(
        f"| **Naive Lexicon Baseline** | {sent['naive_lexicon']['accuracy']:.3f} | {sent['naive_lexicon']['macro_f1']:.3f} | Frequency matching on financial sentiment keywords |"
    )
    md.append(
        f"| **Enhanced VADER** | {sent['vader']['accuracy']:.3f} | {sent['vader']['macro_f1']:.3f} | Rule-based with financial domain lexicon boost |"
    )
    md.append(
        f"| **ProsusAI FinBERT** | **{sent['finbert']['accuracy']:.3f}** | **{sent['finbert']['macro_f1']:.3f}** | Transformer domain-adapted on financial corpora |"
    )
    md.append("\n### FinBERT Confusion Matrix (Rows: True, Cols: Pred: [Neg, Neu, Pos])\n")
    cm = sent["finbert"]["confusion_matrix"]
    md.append(f"```\n{np.array(cm)}\n```\n")
    md.append("![Sentiment Comparison](sentiment_comparison.png)\n")
    md.append("![FinBERT Confusion Matrix](finbert_confusion_matrix.png)\n")

    md.append("\n---\n")
    md.append("## 2. Event Classification Benchmark")
    md.append(
        f"Evaluated on **{ev['dataset_size']} ground-truth labelled financial headlines** across all 12 taxonomy categories.\n"
    )
    md.append("| Classification Architecture | Accuracy | Macro-F1 | Method Description |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(
        f"| **Rules-Only (Lexicon)** | {ev['rules_only']['accuracy']:.3f} | {ev['rules_only']['macro_f1']:.3f} | Weighted keyword regex matching with domain thresholds |"
    )
    md.append(
        f"| **Hybrid (Rules + Zero-Shot)** | **{ev['hybrid']['accuracy']:.3f}** | **{ev['hybrid']['macro_f1']:.3f}** | Ensembled zero-shot NLI + rules arbitration |"
    )

    md.append("\n### Per-Class Performance Breakdown (Hybrid Architecture)\n")
    md.append("| Event Category | Precision | Recall | F1-Score | Support |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for cls_name, vals in ev["hybrid"]["per_class"].items():
        md.append(
            f"| **{cls_name}** | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} | {vals['support']} |"
        )

    md.append("\n### Top Classification Errors\n")
    if ev["common_errors"]:
        md.append("| Headline Excerpt | Ground Truth | Model Prediction |")
        md.append("| :--- | :--- | :--- |")
        for err in ev["common_errors"][:10]:
            clean_txt = err["text"][:75].replace("|", "\\|")
            md.append(f'| "{clean_txt}..." | `{err["true_label"]}` | `{err["predicted_label"]}` |')
    else:
        md.append("No classification errors observed on evaluation dataset.")
    md.append("\n![Events Confusion Matrix](events_confusion_matrix.png)\n")

    md.append("\n---\n")
    md.append("## 3. Market Impact & Sentiment Validation")
    md.append(
        f"Cross-sectional alignment of processed signals from `signals_sample.jsonl` against daily historical closing prices (`yfinance`). Sample size: **{imp['sample_size']} observations**.\n"
    )
    md.append("| Hypothesis Relationship | Spearman Correlation (ρ) | p-value | Statistical Significance |")
    md.append("| :--- | :---: | :---: | :--- |")
    md.append(
        f"| **Impact Score vs Next-Day Absolute Return** | `{imp['impact_vs_abs_return']['spearman_rho']:.4f}` | `{imp['impact_vs_abs_return']['p_value']:.4f}` | {'p < 0.05 (Significant)' if imp['impact_vs_abs_return']['p_value'] < 0.05 else 'p ≥ 0.05 (Not Significant)'} |"
    )
    md.append(
        f"| **Impact Score vs 3-Day Realised Volatility** | `{imp['impact_vs_volatility']['spearman_rho']:.4f}` | `{imp['impact_vs_volatility']['p_value']:.4f}` | {'p < 0.05 (Significant)' if imp['impact_vs_volatility']['p_value'] < 0.05 else 'p ≥ 0.05 (Not Significant)'} |"
    )
    md.append(
        f"| **Sentiment Score vs Signed Next-Day Return** | `{imp['sentiment_vs_return']['spearman_rho']:.4f}` | `{imp['sentiment_vs_return']['p_value']:.4f}` | {'p < 0.05 (Significant)' if imp['sentiment_vs_return']['p_value'] < 0.05 else 'p ≥ 0.05 (Not Significant)'} |"
    )
    md.append(f"\n> **Methodological Disclaimer:** {imp['disclaimer']} Sample size N = {imp['sample_size']}.\n")
    md.append("![Impact Validation](impact_validation_scatter.png)\n")

    md.append("\n---\n")
    md.append("## 4. Pipeline Execution Speed & Throughput")
    md.append(
        "| Pipeline Configuration | Batch Size | Total Latency | Throughput (docs/sec) | Deployment Suitability |"
    )
    md.append("| :--- | :---: | :---: | :---: | :--- |")
    md.append(
        f"| **VADER + Rules Pipeline** | {spd['n_docs']} docs | {spd['vader_rules']['duration_seconds']:.2f}s | **{spd['vader_rules']['docs_per_second']:.1f} docs/s** | Real-time streaming & low-resource CPU |"
    )
    md.append(
        f"| **FinBERT + Rules Pipeline** | {spd['finbert_rules']['processed']} docs | {spd['finbert_rules']['duration_seconds']:.2f}s | **{spd['finbert_rules']['docs_per_second']:.1f} docs/s** | High-precision batch offline processing |"
    )

    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RiskPulse AI/NLP Risk Engine")
    parser.add_argument("--sentiment-samples", type=int, default=1000, help="Number of sentiment samples")
    parser.add_argument("--speed-docs", type=int, default=200, help="Documents for speed benchmark")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("      RISKPULSE ENGINE COMPREHENSIVE EMPIRICAL EVALUATION")
    print("=" * 70 + "\n")

    # A) Sentiment
    sent_metrics = evaluate_sentiment(sample_size=args.sentiment_samples)

    # B) Events
    event_metrics = evaluate_events()

    # C) Impact Validation
    impact_metrics = evaluate_impact_validation()

    # D) Speed
    speed_metrics = evaluate_speed(n_docs=args.speed_docs)

    all_metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sentiment": sent_metrics,
        "events": event_metrics,
        "impact_validation": impact_metrics,
        "speed": speed_metrics,
    }

    # Save metrics.json
    json_path = RESULTS_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved metrics JSON: {json_path}")

    # Generate and save metrics.md
    md_content = generate_markdown_report(all_metrics)
    md_path = RESULTS_DIR / "metrics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved metrics Markdown: {md_path}")

    # Print Summary Table to Terminal
    print("\n" + "=" * 70)
    print("                   EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"1. SENTIMENT ACCURACY & MACRO-F1 (N={sent_metrics['sample_size']}):")
    print(
        f"   - Naive Lexicon: Accuracy={sent_metrics['naive_lexicon']['accuracy']:.3f}, Macro-F1={sent_metrics['naive_lexicon']['macro_f1']:.3f}"
    )
    print(
        f"   - Enhanced VADER: Accuracy={sent_metrics['vader']['accuracy']:.3f}, Macro-F1={sent_metrics['vader']['macro_f1']:.3f}"
    )
    print(
        f"   - ProsusAI FinBERT: Accuracy={sent_metrics['finbert']['accuracy']:.3f}, Macro-F1={sent_metrics['finbert']['macro_f1']:.3f}"
    )

    print(f"\n2. EVENT CLASSIFICATION (N={event_metrics['dataset_size']}):")
    print(
        f"   - Rules-Only: Accuracy={event_metrics['rules_only']['accuracy']:.3f}, Macro-F1={event_metrics['rules_only']['macro_f1']:.3f}"
    )
    print(
        f"   - Hybrid Model: Accuracy={event_metrics['hybrid']['accuracy']:.3f}, Macro-F1={event_metrics['hybrid']['macro_f1']:.3f}"
    )

    print(f"\n3. MARKET IMPACT VALIDATION (N={impact_metrics['sample_size']}):")
    print(
        f"   - Impact Score vs Next-Day |Return|: Spearman rho={impact_metrics['impact_vs_abs_return']['spearman_rho']:.4f} (p={impact_metrics['impact_vs_abs_return']['p_value']:.4f})"
    )
    print(
        f"   - Sentiment Score vs Signed Return: Spearman rho={impact_metrics['sentiment_vs_return']['spearman_rho']:.4f} (p={impact_metrics['sentiment_vs_return']['p_value']:.4f})"
    )
    print(f"   * {impact_metrics['disclaimer']}")

    print("\n4. PIPELINE THROUGHPUT (CPU):")
    print(f"   - VADER+Rules: {speed_metrics['vader_rules']['docs_per_second']:.1f} docs/sec")
    print(f"   - FinBERT+Rules: {speed_metrics['finbert_rules']['docs_per_second']:.1f} docs/sec")
    print("=" * 70 + "\n")
    print(f"All reports and charts saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
