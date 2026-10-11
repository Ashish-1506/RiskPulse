# RiskPulse System Architecture

The following diagram illustrates the end-to-end data processing and inference topology of **RiskPulse**: from multi-source asynchronous data ingestion to NLP enrichment, impact attribution scoring, hybrid storage, REST API serving, downstream risk engines (Module A and Module B), and interactive dashboard visualization.

```mermaid
flowchart TD
    %% 1. Ingestion Sources
    subgraph Sources ["1. Multi-Source Ingestion Layer"]
        S1["GDELT DOC 2.0 Global News API"]
        S2["NewsAPI Everything Financial Stream"]
        S3["RSS Feeds (FT, Reuters, Bloomberg, CNBC)"]
        S4["Social Discourse (StockTwits & Twitter Replay)"]
        S5["Offline Deterministic Sample Replay"]
    end

    %% 2. Ingestion & Normalization
    subgraph Ingestion ["2. Ingestion & Normalization Engine"]
        ING["Unified Normalizer (src/ingestion/normalise.py)"]
        VAL["Validation & Deduplication (Schema: NormalizedDoc)"]
    end

    Sources --> ING
    ING --> VAL

    %% 3. Preprocessing & Entity Recognition
    subgraph Preprocess ["3. Text Processing & Entity Enrichment"]
        CLEAN["Regex Text Sanitizer & HTML Stripper"]
        NER["NER & Ticker Extractor (spacy en_core_web_sm / regex)"]
        MAP["Entity Resolver (GICS Sectors & ISO-2 Countries)"]
    end

    VAL --> CLEAN
    CLEAN --> NER
    NER --> MAP

    %% 4. AI/NLP Inference Layer
    subgraph Inference ["4. AI/NLP Inference Layer (RiskEngine)"]
        direction TB
        SENT["Sentiment Classifier (src/nlp/sentiment.py)<br/>- ProsusAI/finbert (Transformer)<br/>- VADER Financial Fallback"]
        EVENT["12-Class Event Classifier (src/nlp/events.py)<br/>- Rules/Keyword Taxonomy<br/>- Zero-Shot valhalla/distilbart-mnli-12-3"]
    end

    MAP --> SENT
    MAP --> EVENT

    %% 5. Multi-Factor Impact Severity Scorer
    subgraph Impact ["5. Multi-Factor Impact Severity Engine (src/nlp/impact.py)"]
        CORR["Sliding 24h Corroboration Tracker"]
        SCORER["Multi-Factor Scoring Formula<br/>Event Prior + Sentiment + Keywords + Source + Corroboration<br/>Scale: [1.0, 10.0]"]
    end

    SENT --> SCORER
    EVENT --> SCORER
    MAP --> CORR
    CORR --> SCORER

    %% 6. Schema Generation & Hybrid Persistence
    subgraph Storage ["6. Serving & Hybrid Persistence Layer"]
        SCHEMA["Canonical Schema: RiskSignal (Pydantic v2)"]
        SQL[("SQLite Database: data/signals.db<br/>(Signals & Join Indexes)")]
        JSONL[("Append-Only Log: data/signals.jsonl")]
        API["FastAPI REST Service (src/api/main.py)<br/>Port 8000: /signals, /analyze, /health"]
    end

    SCORER --> SCHEMA
    SCHEMA --> SQL
    SCHEMA --> JSONL
    SQL --> API

    %% 7. Downstream Financial Engines
    subgraph Downstream ["7. Downstream Capital & Portfolio Risk Engines"]
        subgraph ModA ["Module A: Tactical Index Rebalancer (src/modules/rebalancer/)"]
            REBAL["Tactical Weight Engine<br/>Exponential Sentiment Decay<br/>Turnover Budget & No-Trade Bands"]
            HIST[("data/weights_history.csv & backtest_curve.csv")]
        end

        subgraph ModB ["Module B: Wholesale Banking Stress Tester (src/modules/stress/)"]
            STRESS["Wholesale Banking Stress Engine<br/>Trigger: Impact Score >= 7.0<br/>Multi-Asset Valuation (Loans, Bonds, Derivs, Equities)<br/>Basel III Delta ECL & Capital Impact"]
            PORT[("data/portfolio.csv ($5.0B Multi-Asset Book)")]
        end
    end

    API -->|High-Frequency Ingestion| REBAL
    API -->|High-Impact Filter (Score >= 7.0)| STRESS
    SQL --> REBAL
    SQL --> STRESS
    REBAL --> HIST
    PORT --> STRESS

    %% 8. Interactive Dashboard & Visualization
    subgraph UI ["8. Interactive Visualizer (Streamlit: Port 8501)"]
        P1["Page 1: Live Risk Signals<br/>(Real-Time Feeds, Filtering & Headline Injection)"]
        P2["Page 2: Tactical Index Rebalancer<br/>(Weights Stack, Replay History & Backtest Equity Curve)"]
        P3["Page 3: Strategic Stress Test<br/>(Book Mark-to-Market, Sector Losers & Reverse Stress Test)"]
    end

    API -.-> UI
    SQL -.-> UI
    HIST -.-> P2
    STRESS -.-> P3
```

---

## Data Flow Narrative

1. **Multi-Source Ingestion**: Asynchronous collectors query live streaming APIs (GDELT DOC 2.0, NewsAPI), monitored financial RSS feeds, and social media data streams. In offline demo mode, high-fidelity sample records (`data/signals_sample.jsonl`) are replayed deterministically.
2. **Normalization & Preprocessing**: Raw documents are cleaned, deduplicated, and mapped into `NormalizedDoc` structures. Regular expressions and NLP extract identified equity tickers (e.g., `AAPL`, `NVDA`, `JPM`), ISO-2 countries, and GICS sectors.
3. **Sentiment & Event Categorization**:
   - Financial sentiment is scored via **ProsusAI FinBERT** (domain-adapted financial transformer) with fallback to rule-enhanced **VADER**.
   - Event categorization maps incoming text into 1 of 12 financial risk taxonomy classes using a hybrid architecture combining rule-based regular expressions and zero-shot NLI (`valhalla/distilbart-mnli-12-3`).
4. **Market Impact Severity Scoring**: Impact is scored on a calibrated scale of `[1.0, 10.0]` factoring in baseline event priors, sentiment severity, high-intensity market keywords, publisher credibility, and cross-source corroboration across a 24-hour sliding window.
5. **Hybrid Persistence & Serving**: Signals are validated against Pydantic schema `RiskSignal`, written to `data/signals.jsonl`, indexed into SQLite (`data/signals.db`), and served via FastAPI REST endpoints (`/signals`, `/analyze`, `/health`).
6. **Downstream Applications**:
   - **Module A (Tactical Index Rebalancing)**: Dynamically tilts a 15-stock S&P 100 benchmark using exponentially decaying sentiment scores under turnover budget and position constraint clamping.
   - **Module B (Wholesale Portfolio Stress Testing)**: When a severe macro or geopolitical signal arrives (`impact_score >= 7.0`), the system executes non-linear multi-asset repricing across a $5B commercial portfolio (loans, bonds, derivatives, equities) to project Mark-to-Market P&L, delta Basel ECL provisions, and capital ratios.
7. **Interactive Dashboard**: A multi-page dark-themed Streamlit application provides live risk monitoring, headline injection, portfolio stress analytics, and historical tactical replay.
