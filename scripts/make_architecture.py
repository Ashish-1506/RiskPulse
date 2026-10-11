"""Generate high-resolution (3840x2160, 16:9) system architecture diagram for RiskPulse.

Uses only matplotlib. Renders 5 clear processing stages corresponding to actual
components in src/, config/, data/, and run_demo.py.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("docs/architecture.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def draw_rounded_box(ax, x, y, w, h, bg="#121829", border="#6366F1", lw=1.5, radius=0.015):
    """Draws a rounded rectangular box with customizable border and background."""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=radius),
        linewidth=lw,
        edgecolor=border,
        facecolor=bg,
        zorder=2,
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#64748B", lw=2.2, label=None, label_offset=(0, 0.018)):
    """Draws a crisp directional arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.45,head_length=0.7",
            color=color,
            lw=lw,
            shrinkA=4,
            shrinkB=4,
            mutation_scale=18,
        ),
        zorder=3,
    )
    if label:
        lx = (x1 + x2) / 2 + label_offset[0]
        ly = (y1 + y2) / 2 + label_offset[1]
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0B0F1A", edgecolor=color, lw=1.0),
            zorder=4,
        )


def build_diagram():
    # 3840 x 2160 px at 200 DPI -> 19.2 x 10.8 inches (16:9)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=200)
    fig.patch.set_facecolor("#0B0F1A")
    ax.set_facecolor("#0B0F1A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Typography defaults
    FONT_SANS = "sans-serif"

    # -------------------------------------------------------------------------
    # Header & Title Block
    # -------------------------------------------------------------------------
    ax.text(
        0.04,
        0.945,
        "RiskPulse: System Architecture and Data Flow",
        fontsize=26,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily=FONT_SANS,
    )
    ax.text(
        0.04,
        0.915,
        "Autonomous AI/NLP Financial Risk Intelligence & Quantitative Stress-Testing Engine",
        fontsize=14,
        color="#94A3B8",
        fontfamily=FONT_SANS,
    )

    # -------------------------------------------------------------------------
    # Column X-Coordinates (5 Pillars, 16:9 Spacing)
    # -------------------------------------------------------------------------
    c_w = 0.165  # Width of standard column box
    c1_x = 0.04  # 1. Data Sources
    c2_x = 0.235  # 2. Ingestion & Preprocessing
    c3_x = 0.430  # 3. AI/NLP RiskEngine
    c4_x = 0.625  # 4. Storage & REST API
    c5_x = 0.815  # 5. Downstream Modules

    # Helper for Stage Badges (Numbered Circles 1-5)
    stage_colors = ["#22D3EE", "#38BDF8", "#6366F1", "#A855F7", "#F59E0B"]
    stage_titles = [
        "1. DATA SOURCES",
        "2. INGEST & PREPROCESS",
        "3. AI/NLP RISK ENGINE",
        "4. SERVING & STORAGE",
        "5. DOWNSTREAM RISK",
    ]
    for i, (sx, scol, stitle) in enumerate(zip([c1_x, c2_x, c3_x, c4_x, c5_x], stage_colors, stage_titles)):
        # Circle badge
        circle = patches.Circle((sx + 0.015, 0.865), 0.011, facecolor=scol, edgecolor="none", zorder=5)
        ax.add_patch(circle)
        ax.text(
            sx + 0.015,
            0.865,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#0B0F1A",
            zorder=6,
        )
        ax.text(
            sx + 0.033,
            0.865,
            stitle,
            va="center",
            fontsize=13,
            fontweight="bold",
            color=scol,
            fontfamily=FONT_SANS,
            zorder=5,
        )

    # =========================================================================
    # COLUMN 1: DATA SOURCES
    # =========================================================================
    draw_rounded_box(ax, c1_x, 0.16, c_w, 0.67, bg="#121829", border="#22D3EE", lw=1.5)

    # Sub-item 1: GDELT
    draw_rounded_box(ax, c1_x + 0.01, 0.675, c_w - 0.02, 0.125, bg="#0E1626", border="#22D3EE", lw=1.0)
    ax.text(c1_x + 0.02, 0.770, "GDELT DOC 2.0 API", fontsize=14, fontweight="bold", color="#22D3EE")
    ax.text(c1_x + 0.02, 0.738, "Global macro & political news", fontsize=11, color="#CBD5E1")
    ax.text(c1_x + 0.02, 0.710, "Tag: News | src/ingestion/gdelt.py", fontsize=9.5, color="#64748B")

    # Sub-item 2: Financial RSS
    draw_rounded_box(ax, c1_x + 0.01, 0.525, c_w - 0.02, 0.125, bg="#0E1626", border="#22D3EE", lw=1.0)
    ax.text(c1_x + 0.02, 0.620, "Public Financial RSS", fontsize=14, fontweight="bold", color="#22D3EE")
    ax.text(c1_x + 0.02, 0.588, "Reuters, FT, Bloomberg, CNBC", fontsize=11, color="#CBD5E1")
    ax.text(c1_x + 0.02, 0.560, "Tag: News | src/ingestion/rss.py", fontsize=9.5, color="#64748B")

    # Sub-item 3: NewsAPI
    draw_rounded_box(ax, c1_x + 0.01, 0.375, c_w - 0.02, 0.125, bg="#0E1626", border="#22D3EE", lw=1.0)
    ax.text(c1_x + 0.02, 0.470, "NewsAPI Everything", fontsize=14, fontweight="bold", color="#22D3EE")
    ax.text(c1_x + 0.02, 0.438, "Live company dispatches", fontsize=11, color="#CBD5E1")
    ax.text(c1_x + 0.02, 0.410, "Tag: News | Optional API key", fontsize=9.5, color="#64748B")

    # Sub-item 4: Social Replay
    draw_rounded_box(ax, c1_x + 0.01, 0.185, c_w - 0.02, 0.165, bg="#0E1626", border="#22D3EE", lw=1.0)
    ax.text(c1_x + 0.02, 0.318, "Social Discourse Stream", fontsize=14, fontweight="bold", color="#22D3EE")
    ax.text(c1_x + 0.02, 0.286, "Kaggle financial tweets replay", fontsize=11, color="#CBD5E1")
    ax.text(c1_x + 0.02, 0.258, "Tag: Social | data/sample_tweets.csv", fontsize=9.5, color="#64748B")
    ax.text(c1_x + 0.02, 0.230, "src/ingestion/tweets_replay.py", fontsize=9.5, color="#64748B")

    # =========================================================================
    # COLUMN 2: INGESTION AND PREPROCESSING
    # =========================================================================
    draw_rounded_box(ax, c2_x, 0.16, c_w, 0.67, bg="#121829", border="#38BDF8", lw=1.5)

    # Sub-item 1: Normalizer & Deduplication
    draw_rounded_box(ax, c2_x + 0.01, 0.615, c_w - 0.02, 0.185, bg="#0E1626", border="#38BDF8", lw=1.0)
    ax.text(c2_x + 0.02, 0.765, "Ingestion & Normalizer", fontsize=14, fontweight="bold", color="#38BDF8")
    ax.text(c2_x + 0.02, 0.735, "NormalizedDoc unified model", fontsize=11, color="#E2E8F0")
    ax.text(c2_x + 0.02, 0.708, "SHA-256 deduplication cache", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.680, "Retry backoff & rate-limits", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.645, "normalise.py | cache.py", fontsize=9.5, color="#64748B")

    # Sub-item 2: Text Preprocessing
    draw_rounded_box(ax, c2_x + 0.01, 0.405, c_w - 0.02, 0.185, bg="#0E1626", border="#38BDF8", lw=1.0)
    ax.text(c2_x + 0.02, 0.555, "Text Sanitization", fontsize=14, fontweight="bold", color="#38BDF8")
    ax.text(c2_x + 0.02, 0.525, "HTML tag & boilerplate strip", fontsize=11, color="#E2E8F0")
    ax.text(c2_x + 0.02, 0.498, "Financial symbol normalization", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.470, "Non-English detection filter", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.435, "src/nlp/preprocess.py", fontsize=9.5, color="#64748B")

    # Sub-item 3: Entity & Taxonomy Extraction
    draw_rounded_box(ax, c2_x + 0.01, 0.185, c_w - 0.02, 0.195, bg="#0E1626", border="#38BDF8", lw=1.0)
    ax.text(c2_x + 0.02, 0.345, "Entity Recognition (NER)", fontsize=14, fontweight="bold", color="#38BDF8")
    ax.text(c2_x + 0.02, 0.315, "15 S&P 100 tickers extraction", fontsize=11, color="#E2E8F0")
    ax.text(c2_x + 0.02, 0.288, "ISO-2 country codes mapping", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.260, "GICS sector classification", fontsize=11, color="#CBD5E1")
    ax.text(c2_x + 0.02, 0.225, "spaCy en_core_web_sm | entities.py", fontsize=9.5, color="#64748B")

    # =========================================================================
    # COLUMN 3: AI / NLP RISK ENGINE (Primary Engine)
    # =========================================================================
    draw_rounded_box(ax, c3_x, 0.16, c_w + 0.01, 0.67, bg="#131B33", border="#6366F1", lw=2.2)

    # Header in Column 3
    ax.text(c3_x + 0.015, 0.800, "RiskEngine (Unified Orchestrator)", fontsize=13, fontweight="bold", color="#A5B4FC")
    ax.text(c3_x + 0.015, 0.780, "src/nlp/engine.py", fontsize=9.5, color="#818CF8")

    # Sub-Box 1: Sentiment Backend
    draw_rounded_box(ax, c3_x + 0.01, 0.630, c_w - 0.01, 0.135, bg="#0E1428", border="#818CF8", lw=1.2)
    ax.text(c3_x + 0.018, 0.735, "Domain Sentiment", fontsize=13, fontweight="bold", color="#A5B4FC")
    ax.text(c3_x + 0.018, 0.708, "ProsusAI/finbert (Transformer)", fontsize=10.5, color="#E0E7FF")
    ax.text(c3_x + 0.018, 0.682, "Rule-enhanced VADER fallback", fontsize=10, color="#C7D2FE")
    ax.text(c3_x + 0.018, 0.655, "Score: [-1.0 .. +1.0] polarity", fontsize=10, color="#38BDF8")

    # Sub-Box 2: Event Taxonomy Classifier
    draw_rounded_box(ax, c3_x + 0.01, 0.475, c_w - 0.01, 0.135, bg="#0E1428", border="#818CF8", lw=1.2)
    ax.text(c3_x + 0.018, 0.580, "Event Classifier", fontsize=13, fontweight="bold", color="#A5B4FC")
    ax.text(c3_x + 0.018, 0.553, "12-class financial taxonomy", fontsize=10.5, color="#E0E7FF")
    ax.text(c3_x + 0.018, 0.527, "Rules + Zero-Shot MNLI (valhalla)", fontsize=10, color="#C7D2FE")
    ax.text(c3_x + 0.018, 0.500, "config/events.yaml | events.py", fontsize=9.5, color="#818CF8")

    # Sub-Box 3: Impact Scorer
    draw_rounded_box(ax, c3_x + 0.01, 0.315, c_w - 0.01, 0.140, bg="#0E1428", border="#818CF8", lw=1.2)
    ax.text(c3_x + 0.018, 0.425, "Multi-Factor Impact Scorer", fontsize=13, fontweight="bold", color="#A5B4FC")
    ax.text(c3_x + 0.018, 0.398, "Event prior + |sentiment| + source", fontsize=10.5, color="#E0E7FF")
    ax.text(c3_x + 0.018, 0.372, "+ High-intensity keywords", fontsize=10, color="#C7D2FE")
    ax.text(c3_x + 0.018, 0.345, "+ 24h sliding corroboration tracker", fontsize=9.5, color="#C7D2FE")
    ax.text(c3_x + 0.018, 0.323, "Calibrated Severity: [1.0 .. 10.0]", fontsize=9.5, color="#F59E0B")

    # Output Chip: RiskSignal Model
    draw_rounded_box(ax, c3_x + 0.008, 0.180, c_w - 0.006, 0.115, bg="#1E1B4B", border="#C084FC", lw=1.5)
    ax.text(c3_x + 0.015, 0.265, "Output Chip: RiskSignal", fontsize=12.5, fontweight="bold", color="#E9D5FF")
    ax.text(c3_x + 0.015, 0.240, "• sentiment_score  [-1.0 .. 1.0]", fontsize=9.5, color="#F3E8FF")
    ax.text(c3_x + 0.015, 0.218, "• event_type (1 of 12 classes)", fontsize=9.5, color="#F3E8FF")
    ax.text(c3_x + 0.015, 0.196, "• impact_score [1.0 .. 10.0]", fontsize=9.5, color="#FCD34D")

    # =========================================================================
    # COLUMN 4: STORAGE AND REST API
    # =========================================================================
    draw_rounded_box(ax, c4_x, 0.16, c_w, 0.67, bg="#121829", border="#A855F7", lw=1.5)

    # Sub-item 1: Dual Persistence
    draw_rounded_box(ax, c4_x + 0.01, 0.595, c_w - 0.02, 0.205, bg="#0E1626", border="#A855F7", lw=1.0)
    ax.text(c4_x + 0.02, 0.765, "Dual Persistence Layer", fontsize=14, fontweight="bold", color="#A855F7")
    ax.text(c4_x + 0.02, 0.735, "SQLite: data/signals.db", fontsize=11, color="#E2E8F0")
    ax.text(c4_x + 0.02, 0.708, "Indexed on event, ticker, impact", fontsize=10.5, color="#CBD5E1")
    ax.text(c4_x + 0.02, 0.680, "JSONL: data/signals.jsonl", fontsize=11, color="#E2E8F0")
    ax.text(c4_x + 0.02, 0.652, "Append-only audit trail", fontsize=10.5, color="#CBD5E1")
    ax.text(c4_x + 0.02, 0.620, "src/storage/store.py", fontsize=9.5, color="#64748B")

    # Sub-item 2: FastAPI Service
    draw_rounded_box(ax, c4_x + 0.01, 0.355, c_w - 0.02, 0.220, bg="#0E1626", border="#A855F7", lw=1.0)
    ax.text(c4_x + 0.02, 0.540, "FastAPI Service (8000)", fontsize=14, fontweight="bold", color="#A855F7")
    ax.text(c4_x + 0.02, 0.510, "GET /health (probe)", fontsize=10.5, color="#C084FC")
    ax.text(c4_x + 0.02, 0.485, "GET /signals (query filters)", fontsize=10.5, color="#C084FC")
    ax.text(c4_x + 0.02, 0.460, "GET /signals/latest", fontsize=10.5, color="#C084FC")
    ax.text(c4_x + 0.02, 0.435, "GET /signals/summary", fontsize=10.5, color="#C084FC")
    ax.text(c4_x + 0.02, 0.410, "POST /analyze (instant NLP)", fontsize=10.5, color="#C084FC")
    ax.text(c4_x + 0.02, 0.380, "Swagger: /docs | main.py", fontsize=9.5, color="#64748B")

    # Sub-item 3: SignalClient
    draw_rounded_box(ax, c4_x + 0.01, 0.185, c_w - 0.02, 0.150, bg="#0E1626", border="#A855F7", lw=1.0)
    ax.text(c4_x + 0.02, 0.305, "SignalClient Adapter", fontsize=14, fontweight="bold", color="#A855F7")
    ax.text(c4_x + 0.02, 0.275, "Transparent HTTP client", fontsize=11, color="#E2E8F0")
    ax.text(c4_x + 0.02, 0.248, "Retry logic + backoff", fontsize=10.5, color="#CBD5E1")
    ax.text(c4_x + 0.02, 0.222, "Direct SQLite fallback", fontsize=10.5, color="#F59E0B")
    ax.text(c4_x + 0.02, 0.198, "src/api/client.py", fontsize=9.5, color="#64748B")

    # =========================================================================
    # COLUMN 5: DOWNSTREAM MODULES
    # =========================================================================
    # Module B: Strategic Wholesale Stress Test (Upper Box)
    draw_rounded_box(ax, c5_x, 0.525, c_w, 0.305, bg="#1E1610", border="#F59E0B", lw=1.8)
    ax.text(c5_x + 0.015, 0.795, "Module B: Stress Test", fontsize=15, fontweight="bold", color="#F59E0B")
    ax.text(c5_x + 0.015, 0.770, "Strategic Wholesale Banking", fontsize=11, color="#FDE68A")
    ax.text(c5_x + 0.015, 0.740, "• $5.02B book: 351 positions", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.718, "  Loans, Bonds, Swaps, Equities", fontsize=10, color="#CBD5E1")
    ax.text(c5_x + 0.015, 0.692, "• Shock scaling by (impact / 10)", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.670, "• Multi-asset pricing valuation:", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.648, "  DV01, CS01, Duration/Convexity", fontsize=10, color="#CBD5E1")
    ax.text(c5_x + 0.015, 0.626, "• Delta ECL credit provisions", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.604, "• Reverse stress testing engine", fontsize=10.5, color="#E2E8F0")
    ax.text(
        c5_x + 0.015, 0.575, "• Automatic trigger: impact >= 7.0", fontsize=10.5, fontweight="bold", color="#EF4444"
    )
    ax.text(c5_x + 0.015, 0.545, "src/modules/stress/ (engine, valuation)", fontsize=9.5, color="#B45309")

    # Module A: Tactical Index Rebalancer (Lower Box)
    draw_rounded_box(ax, c5_x, 0.160, c_w, 0.345, bg="#0D1F18", border="#10B981", lw=1.8)
    ax.text(c5_x + 0.015, 0.470, "Module A: Rebalancer", fontsize=15, fontweight="bold", color="#10B981")
    ax.text(c5_x + 0.015, 0.445, "Tactical Index Allocation", fontsize=11, color="#A7F3D0")
    ax.text(c5_x + 0.015, 0.415, "• 15 S&P 100 stock universe", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.388, "• Time-decayed sentiment scoring", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.362, "• Non-linear tanh tilt mapping", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.336, "• Max weight caps (15% per stock)", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.310, "• 20% turnover budget limit", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.284, "• No-trade friction buffer bands", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.258, "• Backtesting engine (Sharpe, DD)", fontsize=10.5, color="#E2E8F0")
    ax.text(c5_x + 0.015, 0.230, "+4.18% Alpha vs Buy-and-Hold", fontsize=10.5, fontweight="bold", color="#34D399")
    ax.text(c5_x + 0.015, 0.198, "src/modules/rebalancer/ (weights, backtest)", fontsize=9.5, color="#047857")

    # =========================================================================
    # UNIFIED DASHBOARD STRIP (Spanning Across Bottom)
    # =========================================================================
    draw_rounded_box(ax, 0.04, 0.055, 0.94, 0.082, bg="#111827", border="#E2E8F0", lw=1.5)
    ax.text(
        0.06, 0.108, "Streamlit Risk Operations Dashboard (Port 8501)", fontsize=14, fontweight="bold", color="#F8FAFC"
    )
    ax.text(0.06, 0.078, "src/dashboard/app.py | components.py", fontsize=10.5, color="#94A3B8")

    # Dashboard 3 Page Badges
    p1 = patches.FancyBboxPatch(
        (0.35, 0.068),
        0.18,
        0.056,
        boxstyle="round,pad=0.005",
        fc="#1E293B",
        ec="#38BDF8",
        lw=1.0,
        zorder=3,
    )
    ax.add_patch(p1)
    ax.text(
        0.44, 0.103, "Page 1: Live Risk Signals", ha="center", fontsize=11, fontweight="bold", color="#38BDF8", zorder=4
    )
    ax.text(0.44, 0.080, "KPIs, Filters, Headline Injector", ha="center", fontsize=9.5, color="#94A3B8", zorder=4)

    p2 = patches.FancyBboxPatch(
        (0.55, 0.068),
        0.18,
        0.056,
        boxstyle="round,pad=0.005",
        fc="#1E293B",
        ec="#10B981",
        lw=1.0,
        zorder=3,
    )
    ax.add_patch(p2)
    ax.text(
        0.64,
        0.103,
        "Page 2: Tactical Rebalancer",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#10B981",
        zorder=4,
    )
    ax.text(0.64, 0.080, "Allocation Trajectory & Equity Curve", ha="center", fontsize=9.5, color="#94A3B8", zorder=4)

    p3 = patches.FancyBboxPatch(
        (0.75, 0.068),
        0.21,
        0.056,
        boxstyle="round,pad=0.005",
        fc="#1E293B",
        ec="#F59E0B",
        lw=1.0,
        zorder=3,
    )
    ax.add_patch(p3)
    ax.text(
        0.855,
        0.103,
        "Page 3: Strategic Stress Test",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#F59E0B",
        zorder=4,
    )
    ax.text(0.855, 0.080, "Waterfall P&L, ECL & Reverse Stress", ha="center", fontsize=9.5, color="#94A3B8", zorder=4)

    # =========================================================================
    # ARROWS (Clean, Straight Horizontal & Vertical Feed Flow)
    # =========================================================================
    # Col 1 -> Col 2
    draw_arrow(ax, c1_x + c_w, 0.495, c2_x, 0.495, color="#22D3EE", lw=2.4)

    # Col 2 -> Col 3
    draw_arrow(ax, c2_x + c_w, 0.495, c3_x, 0.495, color="#38BDF8", lw=2.4)

    # Col 3 -> Col 4
    draw_arrow(ax, c3_x + c_w + 0.01, 0.495, c4_x, 0.495, color="#818CF8", lw=2.4)

    # Col 4 -> Col 5 (Module B - Triggered on impact >= 7.0)
    draw_arrow(
        ax,
        c4_x + c_w,
        0.675,
        c5_x,
        0.675,
        color="#F59E0B",
        lw=2.4,
        label="impact >= 7.0",
        label_offset=(0, 0.022),
    )

    # Col 4 -> Col 5 (Module A - Continuous Sentiment Decayed Rebalance)
    draw_arrow(
        ax,
        c4_x + c_w,
        0.332,
        c5_x,
        0.332,
        color="#10B981",
        lw=2.4,
        label="signals stream",
        label_offset=(0, 0.022),
    )

    # Col 5 -> Dashboard (Downwards feeds)
    draw_arrow(ax, c5_x + 0.08, 0.160, c5_x + 0.08, 0.138, color="#E2E8F0", lw=2.0)
    draw_arrow(ax, c4_x + 0.08, 0.160, c4_x + 0.08, 0.138, color="#E2E8F0", lw=2.0)

    # =========================================================================
    # FOOTER STRIP
    # =========================================================================
    footer_text = (
        "Offline replay mode: data/signals_sample.jsonl   |   "
        "One-command launcher: python run_demo.py [--offline]   |   "
        "Central parameters: config/*.yaml"
    )
    ax.text(0.50, 0.018, footer_text, ha="center", va="center", fontsize=11, color="#64748B", fontfamily=FONT_SANS)

    plt.tight_layout(pad=0)
    plt.savefig(str(OUTPUT_PATH), facecolor=fig.get_facecolor(), edgecolor="none", dpi=200)
    plt.close()
    print(f"Architecture diagram successfully saved to {OUTPUT_PATH} (3840x2160, 16:9).")


if __name__ == "__main__":
    build_diagram()
