# RiskPulse Portfolio & Data Specifications

This directory contains the underlying datasets, raw historical time series, synthetic wholesale banking portfolios, and serialized risk signals used by the **RiskPulse AI Risk Engine**, **Module A (Tactical Index Rebalancer)**, and **Module B (Strategic Wholesale Stress-Testing Engine)**.

---

## 1. Directory File Inventory & Data Dictionary

| File Path | Rows / Items | Columns / Schema | Source / Generator | Description & License / Usage Note |
| :--- | :---: | :--- | :--- | :--- |
| `data/portfolio.csv` | 351 rows | 23 columns: `position_id`, `asset_class`, `instrument_type`, `counterparty`, `sector`, `country`, `region`, `currency`, `rating`, `notional`, `market_value`, `maturity_years`, `mod_duration`, `convexity`, `spread_duration`, `pd_1y`, `lgd`, `ead`, `dv01`, `cs01`, `fx_ccy`, `beta`, `direction` | Generated via `scripts/make_portfolio.py --seed 42` | Synthetic wholesale banking loan, bond, derivative, and equity book (~$5.02B USD). Synthetic names, no real counterparties. Permissive educational/hackathon use. |
| `data/labelled_events.csv` | 120 rows | 2 columns: `text`, `label` | Expert-curated validation set (10 examples per category across 12 event classes) | Ground-truth benchmark used in `scripts/evaluate.py` to evaluate event classification accuracy, precision, recall, and F1. Permissive MIT license. |
| `data/sample_tweets.csv` | 360 rows | 4 columns: `timestamp`, `ticker`, `text`, `user` | Synthesized via `scripts/generate_sample_tweets.py` | Realistic social discourse simulation covering 15 S&P 100 tickers for offline and replay stream ingestion. Permissive MIT license. |
| `data/signals.jsonl` | 397 rows | JSONL stream conforming to `RiskSignal` (16 attributes) | Output of live/offline ingestion and `RiskEngine.process()` | Append-only historical stream of enriched risk signals. Automatically self-healing and updated during runtime. |
| `data/signals_sample.jsonl` | 388 rows | JSONL stream conforming to `RiskSignal` (16 attributes) | Curated multi-source historical replay snapshot | Deterministic signal snapshot used by `run_demo.py --offline` to ensure zero-network, high-fidelity demo execution. |
| `data/backtest_curve.csv` | 2 rows | 9 columns: `date`, `strategy_equity`, `benchmark_equity`, `net_strategy_equity`, `strategy_return`, `benchmark_return`, `net_strategy_return`, `turnover`, `cost` | Output of `src/modules/rebalancer/backtest.py` | Historical cumulative equity curve comparing sentiment-tilted portfolio vs equal-weight benchmark. |
| `data/weights_history.csv` | 4,290 rows | 5 columns: `timestamp`, `ticker`, `weight`, `delta`, `score` | Output of `IndexRebalancer.rebalance()` | Historical log of tactical asset allocation weights, sentiment scores, and basis-point changes across the 15 S&P 100 constituents. |
| `data/signals.db` | ~456 KB | SQLite tables: `signals`, `signal_tickers` | Managed via `src/storage/store.py` (`SignalStore`) | Indexed relational store supporting low-latency queries by ticker, event type, date range, and impact threshold. |
| `data/sentiment_cache.db` | ~736 KB | SQLite table: `sentiment_cache(text_hash, score, label, backend)` | Managed via `src/nlp/sentiment.py` | Persistent hash-based cache preventing redundant transformer forward passes during re-ingestion and backtesting. |
| `data/raw/sentiment_analysis_for_financial_news.csv` / `all-data.csv` | 4,846 rows | 2 columns: `sentiment`, `text` | Kaggle FinancialPhraseBank (Malo et al., 2014) | Standard financial NLP sentiment dataset with human-annotated sentiment labels (`positive`, `neutral`, `negative`). Used for evaluating sentiment accuracy in `scripts/evaluate.py`. CC-BY-SA 3.0 license. |
| `data/raw/prices_cache.csv` | 127 rows | 16 columns: `Date` + 15 equity ticker adjusted close prices | Cached via `yfinance` daily historical downloads | Historical price series for the 15 S&P 100 constituents (`AAPL`, `AMZN`, `BAC`, `GOOGL`, `JNJ`, `JPM`, `META`, `MSFT`, `NVDA`, `PG`, `TSLA`, `UNH`, `V`, `WMT`, `XOM`) used for impact market correlation and backtesting. |

---

## 2. Wholesale Banking Portfolio Calibration & Assumptions (`data/portfolio.csv`)

The synthetic portfolio models an institutional wholesale banking book of **~$5.02 Billion USD** in total mark-to-market value across 351 credit facilities, capital markets issues, and derivative hedges.

### Asset Class Composition
| Asset Class | Target Weight | Actual Market Value | Position Count | Instruments Represented |
| :--- | :---: | :---: | :---: | :--- |
| **Loans** | ~45% | $2,263,300,718.51 (45.1%) | 101 | Term Loans, Revolving Credit Facilities (Revolvers), Trade Finance |
| **Bonds** | ~30% | $1,503,551,502.89 (30.0%) | 71 | Corporate Bonds, Sovereign Debt |
| **Derivatives** | ~18% | $900,000,000.00 (17.9%) | 146 | Interest Rate Swaps (IRS), FX Forwards, Credit Default Swaps (CDS) |
| **Equity** | ~7% | $350,000,000.00 (7.0%) | 33 | Strategic Equity Holdings |
| **Total** | **100%** | **$5,016,852,221.40** | **351** | Multi-asset commercial wholesale banking portfolio |

### Portfolio Risk Schema Details
Each position contains the following calibrated parameters:
- `position_id`: Unique identifier (`POS_0001` through `POS_0351`).
- `asset_class`: Asset class bucket (`Loan`, `Bond`, `Derivative`, `Equity`).
- `instrument_type`: Financial product classification (`Term Loan`, `Revolver`, `Trade Finance`, `Corporate Bond`, `Sovereign Bond`, `IRS`, `FX Forward`, `CDS`, `Equity Holding`).
- `counterparty`: Fictitious institutional counterparty (`Counterparty_001` through `Counterparty_150`, or sovereign obligor `Sovereign_<Country>`). No real company names are used.
- `sector`: Industry sector (`Energy`, `Financials`, `Technology`, `Healthcare`, `Industrials`, `Utilities`, `Consumer`, `Real Estate`).
- `country`: Domicile country code (`US`, `UK`, `DE`, `IN`, `CN`, `BR`, `AE`, `TR`, `ZA`, `JP`, `MX`).
- `region`: Geographic macro region (`North America`, `Europe`, `Asia-Pacific`, `Latin America`, `Middle East`, `EMEA`, `Africa`).
- `currency`: Contractual denomination (`USD`, `EUR`, `GBP`, `JPY`, `INR`, `CNY`, `BRL`, `AED`, `TRY`, `ZAR`, `MXN`).
- `rating`: Credit rating (`AAA`, `AA`, `A`, `BBB`, `BB`, `B`, `CCC`).
- `notional`: Face principal value (in USD).
- `market_value`: Current mark-to-market valuation (in USD).
- `maturity_years`: Tenor to maturity (0.25 to 12.0 years; 0.0 for equities).
- `mod_duration`: Modified interest rate duration ($\approx 0.85 \times \text{tenor}$ for fixed bonds; $\approx 0.25$ for floating loans; $\approx 0.70 \times \text{tenor}$ for IRS).
- `convexity`: Second-order rate sensitivity ($\approx 0.12 \times (\text{mod\_duration})^2$ for fixed debt; $\approx 0.01$ for floating facilities).
- `spread_duration`: Credit spread duration ($\approx 0.85 \times \text{tenor}$ for corporate credit; $\approx 0.95 \times \text{tenor}$ for CDS).
- `pd_1y`: Basel 1-year annualized Probability of Default calibrated to S&P empirical migration curves (AAA: 0.01%, AA: 0.05%, A: 0.10%, BBB: 0.50%, BB: 2.00%, B: 6.50%, CCC: 20.00%).
- `lgd`: Loss Given Default based on instrument seniority (Sovereign: 25%, Trade Finance: 30%, Term Loans: 40%, Revolvers: 45%, Corporate Bonds: 50%, Derivatives: 55%, Equity: 100%).
- `ead`: Exposure at Default (Loans/Bonds: Market Value; Revolvers: Drawn + Undrawn Credit Conversion Factor $0.75 \times \text{Notional}$; Derivatives: $\max(0, \text{MtM}) + 0.05 \times \text{Notional}$).
- `dv01`: Dollar Value of a Basis Point ($\text{Notional} \times \text{mod\_duration} \times 0.0001 \times \text{direction}$).
- `cs01`: Credit Spread 01 sensitivity ($\text{Notional} \times \text{spread\_duration} \times 0.0001 \times \text{direction}$).
- `fx_ccy`: Foreign currency leg subjected to FX depreciation shocks against USD.
- `beta`: Systematic equity beta relative to global indices ($0.7$ to $1.6$ for equities/corporates; $0.0$ for sovereign debt).
- `direction`: Position sign ($+1$ for Long / receive fixed / sell protection; $-1$ for Short / pay fixed / buy protection).

### Concentration Risk Clusters
To produce realistic wholesale stress outcomes under geopolitical, macroeconomic, and credit events, the book contains realistic geographic and sector clusters:
1. **Middle Eastern Energy Cluster:** ~$1.12B USD in Gulf energy facilities, state-linked debt, and project finance sensitive to commodity price surges and regional conflict.
2. **Asian Real Estate Cluster:** ~$860M USD in commercial property facilities (ratings BB through CCC) with high spread sensitivity and elevated default probability.
3. **North American Technology Cluster:** ~$790M USD in high-beta equity positions, corporate bonds, and interest rate swap overlays.
4. **European Financials Cluster:** Cross-currency debt and interest rate derivatives with financial counterparties.

---

## 3. Data Reproduction & Download Instructions

### Downloading Full External Datasets
1. **Kaggle FinancialPhraseBank Dataset**:
   - URL: [Kaggle: Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
   - Download the file `all-data.csv` and place it in `data/raw/sentiment_analysis_for_financial_news.csv`.
2. **Historical Market Prices**:
   - Executing `python scripts/evaluate.py` will automatically query `yfinance` for the 15 S&P 100 constituents and cache daily bars into `data/raw/prices_cache.csv`.

### Re-Generating Synthetic Portfolios and Datasets
```bash
# Re-generate synthetic wholesale banking portfolio (seed=42)
python scripts/make_portfolio.py --seed 42 --out data/portfolio.csv

# Re-generate synthetic social media discourse dataset
python scripts/generate_sample_tweets.py

# Re-run empirical evaluation suite across sentiment, events, and market validation
python scripts/evaluate.py
```
