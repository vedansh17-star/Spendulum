"""
Spendulum — Complete Financial OS.
Preserving approved visual design & WebGL Prism theme while powering all 8 connected pages,
clickable interactive cards, persistent SQLite storage, demo finance profile, and hybrid AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import importlib
import sys
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from datetime import date, datetime
import os
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

# Import persistent database layer
import db

# =========================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =========================================================

st.set_page_config(
    page_title="Spendulum — Financial Command Center",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database tables and seed demo profile if not already done
db.init_db()
db.seed_demo_profile()

PROJECT_ROOT = Path(__file__).resolve().parent
ML_SOURCE_DIR = PROJECT_ROOT / "ml" / "src"

SUPPORTED_CATEGORIES = [
    "Housing and utilities",
    "Food and groceries",
    "Transport and fuel",
    "Healthcare",
    "Shopping",
    "Entertainment",
    "Education",
    "Travel",
    "Other expenses",
]

CATEGORY_PALETTE = {
    "Housing and utilities": "#104B34",
    "Food and groceries": "#176B45",
    "Transport and fuel": "#F2C94C",
    "Healthcare": "#059669",
    "Shopping": "#2EAF70",
    "Entertainment": "#F5A623",
    "Education": "#047857",
    "Travel": "#10B981",
    "Other expenses": "#68756C",
}

# =========================================================
# 2. SESSION NAVIGATION STATE (CLICKABLE CARDS ENGINE)
# =========================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

if "transaction_filter_category" not in st.session_state:
    st.session_state.transaction_filter_category = "All"

if "transaction_filter_type" not in st.session_state:
    st.session_state.transaction_filter_type = "All"

def set_page(page_name, filter_cat="All", filter_type="All"):
    st.session_state.current_page = page_name
    st.session_state.transaction_filter_category = filter_cat
    st.session_state.transaction_filter_type = filter_type

# =========================================================
# 3. APPROVED STYLING & REACT BITS PRO FINTECH CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --color-emerald-950: #062317;
        --color-emerald-900: #0a3825;
        --color-emerald-800: #104B34;
        --color-emerald-700: #176B45;
        --color-emerald-500: #22C55E;
        --color-emerald-400: #4ADE80;
        --color-gold-400: #F2C94C;
        --color-gold-500: #EAB308;
        --glass-bg: rgba(255, 255, 255, 0.78);
        --glass-card: rgba(255, 255, 255, 0.88);
        --glass-border: rgba(23, 107, 69, 0.12);
        --glass-shadow: 0 10px 30px -5px rgba(16, 75, 52, 0.08), 0 4px 12px -2px rgba(16, 75, 52, 0.04);
        --glass-hover-shadow: 0 20px 40px -10px rgba(23, 107, 69, 0.16);
    }

    /* Base Body & Canvas - Transparent to let the animated WebGL Prism shine through */
    .stApp {
        background: transparent !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F1F17;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    /* Sidebar - Premium Deep Forest Glass */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07261a 0%, #0d3826 60%, #082117 100%) !important;
        border-right: 1px solid rgba(242, 201, 76, 0.15);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.15);
    }
    section[data-testid="stSidebar"] * {
        color: #E6F4ED !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #D1E7DD !important;
        font-weight: 500;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(242, 201, 76, 0.12) !important;
        color: #F2C94C !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] input:checked + div {
        background: #F2C94C !important;
        box-shadow: 0 0 12px rgba(242, 201, 76, 0.8) !important;
    }

    /* React Bits Pro "Navbar 7" Two-Tier Header Replica */
    .pro-navbar {
        background: var(--glass-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 16px 24px;
        margin-bottom: 24px;
        box-shadow: var(--glass-shadow);
        display: flex;
        flex-direction: column;
        gap: 12px;
        position: relative;
        overflow: hidden;
    }
    .pro-navbar::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #176B45 0%, #F2C94C 50%, #22C55E 100%);
    }
    .pro-navbar-tier1 {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(23, 107, 69, 0.08);
        padding-bottom: 10px;
    }
    .pro-navbar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .pro-navbar-badge {
        background: linear-gradient(135deg, #176B45 0%, #104B34 100%);
        color: #F2C94C;
        font-weight: 800;
        font-size: 11px;
        letter-spacing: 0.08em;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        box-shadow: 0 2px 8px rgba(23, 107, 69, 0.25);
    }
    .pro-navbar-status {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        font-weight: 600;
        color: #176B45;
    }
    .status-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
        animation: pulseGreen 2s infinite;
    }
    @keyframes pulseGreen {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }
    .pro-navbar-tier2 {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .pro-navbar-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0d3826;
        margin: 0;
    }
    .pro-navbar-desc {
        font-size: 13px;
        color: #4A6355;
        margin: 0;
    }
    .pro-navbar-kpis {
        display: flex;
        gap: 12px;
    }
    .pro-kpi-pill {
        background: rgba(23, 107, 69, 0.06);
        border: 1px solid rgba(23, 107, 69, 0.12);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
        color: #104B34;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Metric Cards with Clickable Elevation & React Bits SpotlightCard Hover */
    .metric-click-card {
        background: var(--glass-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: var(--glass-shadow);
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
    }
    .metric-click-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: radial-gradient(320px circle at var(--spotlight-x, 50%) var(--spotlight-y, 0%), rgba(23, 107, 69, 0.08), transparent 75%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
    }
    .metric-click-card:hover::before {
        opacity: 1;
    }
    .metric-click-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--glass-hover-shadow);
        border-color: #F2C94C;
    }
    .card-emerald {
        background: linear-gradient(135deg, #104B34 0%, #176B45 60%, #1f8557 100%) !important;
        border: 1px solid rgba(242, 201, 76, 0.3) !important;
    }
    .card-emerald::before {
        background: radial-gradient(320px circle at var(--spotlight-x, 50%) var(--spotlight-y, 0%), rgba(242, 201, 76, 0.16), transparent 75%) !important;
    }
    .card-emerald * {
        color: #FFFFFF !important;
    }
    .mc-sub {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #68756C;
        margin-bottom: 4px;
    }
    .mc-val {
        font-size: 26px;
        font-weight: 800;
        color: #0A2E1F;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .mc-hint {
        font-size: 11px;
        font-weight: 600;
        color: #176B45;
    }

    /* React Bits <PillNav /> Navigation */
    .react-bits-pill-nav {
        background: rgba(255, 255, 255, 0.78);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(23, 107, 69, 0.16);
        border-radius: 9999px;
        padding: 4px 6px;
        box-shadow: 0 4px 18px rgba(16, 75, 52, 0.05);
        margin-bottom: 16px;
    }
    .react-bits-pill-nav .stButton button {
        border-radius: 9999px !important;
        border: 1px solid transparent !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        padding: 6px 12px !important;
        background: transparent !important;
        color: #104B34 !important;
        box-shadow: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .react-bits-pill-nav .stButton button:hover {
        background: rgba(23, 107, 69, 0.08) !important;
        color: #176B45 !important;
        transform: translateY(-1px) !important;
    }
    .react-bits-pill-nav .stButton button[kind="primary"] {
        background: #176B45 !important;
        color: #F2C94C !important;
        box-shadow: 0 4px 14px rgba(23, 107, 69, 0.3) !important;
        transform: scale(1.02) !important;
    }

    /* React Bits <FadeContent /> */
    .react-bits-fade-content {
        animation: rbFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes rbFadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Container Glass Boxes */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--glass-card) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        box-shadow: var(--glass-shadow) !important;
    }

    /* High Contrast Navigation Buttons */
    .stButton button {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #0A2E1F !important;
        border: 1px solid rgba(23, 107, 69, 0.25) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton button:hover {
        background: #176B45 !important;
        color: #FFFFFF !important;
        border-color: #104B34 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(23, 107, 69, 0.25) !important;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #176B45 0%, #104B34 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(23, 107, 69, 0.25) !important;
    }

    /* Tabs Styling - High Contrast Visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: transparent !important;
        border-bottom: 2px solid rgba(23, 107, 69, 0.15) !important;
        margin-bottom: 16px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.8) !important;
        color: #0A2E1F !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid rgba(23, 107, 69, 0.15) !important;
        border-bottom: none !important;
        padding: 8px 18px !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #176B45 !important;
        background: #FFFFFF !important;
    }
    .stTabs [aria-selected="true"] {
        background: #176B45 !important;
        color: #FFFFFF !important;
        border-color: #176B45 !important;
    }

    /* Interactive Top Nav Pills in Navbar */
    .nav-pill-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 8px;
        padding-top: 10px;
        border-top: 1px solid rgba(23, 107, 69, 0.08);
    }

    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. ML CLASSIFIER CACHE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_hybrid_classifier():
    if str(ML_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(ML_SOURCE_DIR))
    return importlib.import_module("hybrid_classifier")

@st.cache_resource(show_spinner=False)
def load_feedback_writer():
    if str(ML_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(ML_SOURCE_DIR))
    return importlib.import_module("user_feedback")

def classify_record(description):
    classifier = load_hybrid_classifier()
    return classifier.predict_category(description)

def teach_correction(description, predicted_cat, corrected_cat):
    classifier = load_hybrid_classifier()
    cleaned = classifier.clean_transaction(description)
    load_feedback_writer().save_feedback(
        description=description,
        cleaned_description=cleaned,
        predicted_category=predicted_cat,
        corrected_category=corrected_cat
    )
    classifier.feedback_rules[cleaned] = corrected_cat

def fmt_inr(val):
    return f"₹{val:,.0f}"

# =========================================================
# 5. SIDEBAR NAVIGATION CONTROLLER
# =========================================================

nav_options = [
    "Overview",
    "Transactions",
    "AI Categorization",
    "Review Queue",
    "Spending Analytics",
    "Budgets",
    "Income & Cash Flow",
    "Settings"
]

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 4px 18px 4px;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:26px;">💸</span>
                <div>
                    <h2 style="font-size:20px; font-weight:800; color:#F2C94C; margin:0; line-height:1;">
                        Spendulum
                    </h2>
                    <span style="font-size:11px; color:#A8D4BC; font-weight:600; letter-spacing:0.05em;">
                        FINANCIAL COMMAND CENTER
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Sync radio selection with st.session_state.current_page
    nav_index = nav_options.index(st.session_state.current_page) if st.session_state.current_page in nav_options else 0
    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=nav_index,
        label_visibility="collapsed",
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    st.markdown("---")

    # Real-time summary widget
    txns_all = db.get_all_transactions()
    unresolved_count = len(txns_all[txns_all["confidence"] == "Low"]) if not txns_all.empty else 0
    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.06); padding:14px; border-radius:12px; border:1px solid rgba(242,201,76,0.2);">
            <div style="font-size:11px; color:#A8D4BC; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;">
                Database Health
            </div>
            <div style="font-size:18px; font-weight:800; color:#F2C94C; margin:4px 0 2px 0;">
                {len(txns_all):,} Records
            </div>
            <div style="font-size:11px; color:{'#F87171' if unresolved_count > 0 else '#4ADE80'};">
                ● {unresolved_count} pending reviews
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Current active page
active_page = st.session_state.current_page

# Navbar Component with Integrated Top-Bar Page Navigation (React Bits Pro Inspired)
def render_header(title, subtitle):
    txns = db.get_all_transactions()
    total_cnt = len(txns)
    rev_cnt = len(txns[txns["confidence"] == "Low"]) if not txns.empty else 0
    st.markdown(
        f"""
        <div class="pro-navbar">
            <div class="pro-navbar-tier1">
                <div class="pro-navbar-brand">
                    <span style="font-size:20px;">💸</span>
                    <span style="font-weight:800; font-size:17px; letter-spacing:-0.03em; color:#0A2E1F;">
                        SPENDULUM <span style="color:#176B45;">PRO</span>
                    </span>
                    <span class="pro-navbar-badge">AI FINANCIAL OS</span>
                </div>
                <div class="pro-navbar-status">
                    <div class="status-pulse"></div>
                    <span>Active SQLite Engine</span>
                    <span style="color:rgba(23,107,69,0.25); margin:0 4px;">|</span>
                    <span style="font-size:12px; color:#68756C;">{datetime.now().strftime('%d %b %Y, %I:%M %p')}</span>
                </div>
            </div>
            <div class="pro-navbar-tier2">
                <div>
                    <h1 class="pro-navbar-title">{title}</h1>
                    <p class="pro-navbar-desc">{subtitle}</p>
                </div>
                <div class="pro-navbar-kpis">
                    <div class="pro-kpi-pill">📊 Stored: <strong>{total_cnt:,}</strong></div>
                    <div class="pro-kpi-pill" style="{'background:rgba(220,38,38,0.1); color:#991B1B;' if rev_cnt else ''}">
                        ⚠️ Needs Review: <strong>{rev_cnt}</strong>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # React Bits <PillNav /> Navigation Bar
    st.markdown('<div class="react-bits-pill-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(8)
    for idx, page_name in enumerate(nav_options):
        is_active = (page_name == st.session_state.current_page)
        with nav_cols[idx]:
            label = f"● {page_name}" if is_active else page_name
            if st.button(label, key=f"top_nav_{page_name}_{idx}", use_container_width=True, type="primary" if is_active else "secondary"):
                set_page(page_name)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)


# =========================================================
# 6. PAGE 1: FINANCIAL OVERVIEW (CLICKABLE METRICS)
# =========================================================

if active_page == "Overview":
    render_header("Financial Overview", "Real-time command center. Click any card or indicator to navigate to its full report.")

    st.markdown(
        """
        <div style="background:rgba(23,107,69,0.06); border:1px solid rgba(23,107,69,0.18); border-left:4px solid #176B45; border-radius:10px; padding:10px 16px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:16px;">🛡️</span>
                <span style="font-size:12px; font-weight:700; color:#0A2E1F;">
                    DEMONSTRATION DATASET ACTIVE &bull; Unified baseline: ₹11,50,000 / year (~₹95,833/month salary across Aug, Sep, Oct 2026). All metrics, budgets, ML models, and review queue items derive live from this single SQLite database.
                </span>
            </div>
            <span style="background:#176B45; color:#F2C94C; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:800; white-space:nowrap;">
                48 Records Synced
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    txns = db.get_all_transactions()
    expenses = txns[txns["type"] == "Expense"].copy()
    incomes = txns[txns["type"] == "Income"].copy()

    total_records = len(txns)
    review_needed = len(txns[txns["confidence"] == "Low"]) if not txns.empty else 0
    total_exp = expenses["amount"].sum()
    total_inc = incomes["amount"].sum()
    net_flow = total_inc - total_exp
    avg_ticket = expenses["amount"].mean() if not expenses.empty else 0.0

    # Row 1: Clickable Metric Cards with Real Callback Navigation
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-click-card">
                <div class="mc-sub">Total Records</div>
                <div class="mc-val">{total_records:,}</div>
                <div class="mc-hint">Click below to view all &rarr;</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Transactions", key="btn_nav_txns", use_container_width=True):
            set_page("Transactions")
            st.rerun()

    with c2:
        st.markdown(
            f"""
            <div class="metric-click-card" style="{'border-color:#DC2626;' if review_needed else ''}">
                <div class="mc-sub" style="{'color:#DC2626;' if review_needed else ''}">Needs Review</div>
                <div class="mc-val" style="{'color:#DC2626;' if review_needed else ''}">{review_needed}</div>
                <div class="mc-hint">Click to resolve items &rarr;</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Review Queue", key="btn_nav_review", use_container_width=True):
            set_page("Review Queue")
            st.rerun()

    with c3:
        st.markdown(
            f"""
            <div class="metric-click-card">
                <div class="mc-sub">Total Expenses</div>
                <div class="mc-val">{fmt_inr(total_exp)}</div>
                <div class="mc-hint">Click for spending analysis &rarr;</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Analytics", key="btn_nav_analytics", use_container_width=True):
            set_page("Spending Analytics")
            st.rerun()

    with c4:
        is_pos = net_flow >= 0
        st.markdown(
            f"""
            <div class="metric-click-card card-emerald">
                <div class="mc-sub">Net Cash Flow</div>
                <div class="mc-val" id="headline-net-flow" data-val="{int(abs(net_flow)) if is_pos else -int(abs(net_flow))}" style="color:#F2C94C !important;">{'+' if is_pos else '-'}{fmt_inr(abs(net_flow))}</div>
                <div class="mc-hint" style="color:#D1E7DD !important;">Click for income summary &rarr;</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Income & Cash Flow", key="btn_nav_cashflow", use_container_width=True):
            set_page("Income & Cash Flow")
            st.rerun()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Row 2: Secondary Quick Cards with Click Navigation
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            f"""
            <div class="metric-click-card" style="padding:14px 20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="mc-sub">Total Income Recorded</div>
                        <div style="font-size:22px; font-weight:800; color:#15803D;">{fmt_inr(total_inc)}</div>
                    </div>
                    <div style="font-size:12px; font-weight:700; color:#15803D;">Salary Baseline: ₹95,833/mo</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Explore Income Breakdown &rarr;", key="btn_nav_inc_quick", use_container_width=True):
            set_page("Income & Cash Flow")
            st.rerun()

    with sc2:
        st.markdown(
            f"""
            <div class="metric-click-card" style="padding:14px 20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="mc-sub">Average Outflow Ticket</div>
                        <div style="font-size:22px; font-weight:800; color:#0A2E1F;">{fmt_inr(avg_ticket)}</div>
                    </div>
                    <div style="font-size:12px; font-weight:600; color:#68756C;">Per Debit Transaction</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Explore Outflow Breakdown &rarr;", key="btn_nav_exp_quick", use_container_width=True):
            set_page("Spending Analytics")
            st.rerun()

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Charts Grid
    ch1, ch2 = st.columns([7, 3])
    with ch1:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 8px 0; font-size:16px; font-weight:800; color:#0A2E1F;'>📈 Monthly Expenditure Trajectory</h4>", unsafe_allow_html=True)
            if not expenses.empty and expenses["date"].notna().any():
                expenses["Month"] = expenses["date"].dt.strftime("%b %Y")
                # Group by Month and Category
                monthly_totals = expenses.groupby(["Month", "category"])["amount"].sum().reset_index()
                # Ensure chronological order
                months_order = expenses.sort_values("date")["Month"].unique().tolist()

                fig = px.bar(
                    monthly_totals,
                    x="Month",
                    y="amount",
                    color="category",
                    category_orders={"Month": months_order},
                    color_discrete_map=CATEGORY_PALETTE,
                    barmode="stack",
                    labels={"amount": "Expenses (₹)", "category": "Category", "Month": "Billing Period"}
                )
                fig.update_layout(
                    height=280,
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.15, x=1, xanchor="right", font=dict(size=10)),
                    xaxis=dict(showgrid=True, gridcolor="rgba(23,107,69,0.06)", tickfont=dict(color="#0A2E1F", weight="bold")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(23,107,69,0.08)", tickprefix="₹", tickfont=dict(color="#0A2E1F"))
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                if st.button("Deep Dive in Spending Analytics ↗", key="btn_chart_analytics", use_container_width=True):
                    set_page("Spending Analytics")
                    st.rerun()
            else:
                st.info("No dated expense records.")

    with ch2:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 8px 0; font-size:16px; font-weight:800;'>🥧 Category Allocations</h4>", unsafe_allow_html=True)
            if not expenses.empty:
                cat_sum = expenses.groupby("category")["amount"].sum().reset_index()
                fig_donut = px.pie(cat_sum, names="category", values="amount", hole=0.6, color="category", color_discrete_map=CATEGORY_PALETTE)
                fig_donut.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
                if st.button("Configure Category Budgets ↗", key="btn_chart_budgets", use_container_width=True):
                    set_page("Budgets")
                    st.rerun()
            else:
                st.info("No expense data.")

    # React Bits <FadeContent />: Below-the-fold Insights
    st.markdown('<div class="react-bits-fade-content">', unsafe_allow_html=True)
    g1, g2 = st.columns([6, 4])
    with g1:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:15px; font-weight:700;'>Recent Transactions</h4>", unsafe_allow_html=True)
            if not txns.empty:
                preview_txns = txns.head(6).copy()
                preview_txns["Formatted Date"] = preview_txns["date"].dt.strftime("%d %b %Y").fillna("—")
                preview_txns["Formatted Amount"] = preview_txns.apply(lambda r: f"{'-' if r['type']=='Expense' else '+'}{fmt_inr(r['amount'])}", axis=1)
                st.dataframe(preview_txns[["Formatted Date", "description", "Formatted Amount", "category", "confidence"]].rename(
                    columns={"Formatted Date": "Date", "description": "Description", "Formatted Amount": "Amount"}
                ), use_container_width=True, hide_index=True)
                if st.button("Browse & Manage All Transactions ↗", key="btn_view_all_txns", use_container_width=True):
                    set_page("Transactions")
                    st.rerun()
            else:
                st.caption("No records.")

    with g2:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0; font-size:15px; font-weight:700;'>Spending Signals & Budget Alerts</h4>", unsafe_allow_html=True)
            budgets = db.get_budgets()
            tot_b = budgets.get("_monthly_total", 61000.0)
            
            # Latest month calculation
            if not expenses.empty:
                latest_m = expenses["date"].dt.to_period("M").max().strftime("%Y-%m")
                m_exp = expenses[expenses["date"].dt.strftime("%Y-%m") == latest_m]["amount"].sum()
                st.markdown(
                    f"""
                    <div style="background:#F0F8F4; border-left:4px solid #176B45; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <strong style="color:#104B34;">Monthly Budget Tracking ({latest_m})</strong>
                        <p style="margin:4px 0 0 0; font-size:13px;">Spent <strong>{fmt_inr(m_exp)}</strong> of <strong>{fmt_inr(tot_b)}</strong> budget limit.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if review_needed > 0:
                st.markdown(
                    f"""
                    <div style="background:#FFF8E8; border-left:4px solid #F2C94C; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <strong style="color:#854D0E;">{review_needed} Unresolved Low-Confidence Item(s)</strong>
                        <p style="margin:4px 0 0 0; font-size:13px;">Open the Review Queue to confirm categories and improve your hybrid AI accuracy.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Resolve Review Items Now ↗", key="btn_resolve_rq", use_container_width=True):
                    set_page("Review Queue")
                    st.rerun()
            if st.button("Monitor Category Budget Ceilings ↗", key="btn_monitor_budgets", use_container_width=True):
                set_page("Budgets")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 7. PAGE 2: TRANSACTIONS (FULL MANAGEMENT)
# =========================================================

elif active_page == "Transactions":
    render_header("Transactions", "Explore, search, filter, edit, delete, and batch import financial records.")

    t_list, t_add, t_import = st.tabs(["  📋 Record Browser  ", "  ➕ Add Single Entry  ", "  📂 CSV Statement Import  "])

    with t_list:
        txns = db.get_all_transactions()
        if txns.empty:
            st.info("No transactions found.")
        else:
            with st.container(border=True):
                fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 2])
                with fc1:
                    s_query = st.text_input("Search description or merchant", placeholder="Swiggy, DMart, Rent...")
                with fc2:
                    all_cats = ["All"] + SUPPORTED_CATEGORIES
                    c_idx = all_cats.index(st.session_state.transaction_filter_category) if st.session_state.transaction_filter_category in all_cats else 0
                    s_cat = st.selectbox("Category", all_cats, index=c_idx)
                with fc3:
                    type_opts = ["All", "Expense", "Income"]
                    t_idx = type_opts.index(st.session_state.transaction_filter_type) if st.session_state.transaction_filter_type in type_opts else 0
                    s_type = st.selectbox("Type", type_opts, index=t_idx)
                with fc4:
                    s_conf = st.selectbox("Confidence", ["All", "High", "Medium", "Low"])

            filtered = txns.copy()
            if s_query:
                filtered = filtered[filtered["description"].str.contains(s_query, case=False, na=False)]
            if s_cat != "All":
                filtered = filtered[filtered["category"] == s_cat]
            if s_type != "All":
                filtered = filtered[filtered["type"] == s_type]
            if s_conf != "All":
                filtered = filtered[filtered["confidence"] == s_conf]

            st.write(f"Showing **{len(filtered):,}** records matching filters:")

            disp = filtered.copy()
            disp["Formatted Date"] = disp["date"].dt.strftime("%d %b %Y").fillna("—")
            disp["Formatted Amount"] = disp.apply(lambda r: f"{'-' if r['type']=='Expense' else '+'}{fmt_inr(r['amount'])}", axis=1)

            st.dataframe(
                disp[["Formatted Date", "description", "Formatted Amount", "type", "category", "confidence", "prediction_source", "review_status"]].rename(
                    columns={"Formatted Date": "Date", "description": "Description", "Formatted Amount": "Amount", "prediction_source": "Source"}
                ),
                use_container_width=True,
                hide_index=True
            )

            # Edit & Delete Action Box
            with st.expander("✏️ Edit or Delete Record"):
                rec_map = {f"{r['description'][:30]} | {fmt_inr(r['amount'])} | {r['date'].strftime('%d %b %Y') if pd.notna(r['date']) else ''}": r["transaction_id"] for _, r in filtered.head(40).iterrows()}
                if rec_map:
                    pick_label = st.selectbox("Select record", list(rec_map.keys()))
                    pick_id = rec_map[pick_label]
                    rec = filtered[filtered["transaction_id"] == pick_id].iloc[0]

                    ed1, ed2 = st.columns([3, 1])
                    with ed1:
                        with st.form("edit_rec_form"):
                            st.write(f"**Modifying:** {rec['description']}")
                            e_desc = st.text_input("Description", value=rec["description"])
                            e_amt = st.number_input("Amount (₹)", value=float(rec["amount"]), min_value=0.0, step=1.0)
                            e_cat = st.selectbox("Category", SUPPORTED_CATEGORIES, index=SUPPORTED_CATEGORIES.index(rec["category"]) if rec["category"] in SUPPORTED_CATEGORIES else 0)
                            e_type = st.selectbox("Type", ["Expense", "Income"], index=0 if rec["type"] == "Expense" else 1)
                            if st.form_submit_button("Save Edits", type="primary"):
                                db.update_transaction(pick_id, {
                                    "description": e_desc.strip(),
                                    "amount": e_amt,
                                    "category": e_cat,
                                    "type": e_type
                                })
                                st.success("Record updated successfully.")
                                st.rerun()
                    with ed2:
                        st.write("**Danger Zone**")
                        c_del = st.checkbox("Confirm Deletion", key=f"chk_del_{pick_id}")
                        if c_del and st.button("Delete Transaction", type="primary"):
                            db.delete_transaction(pick_id)
                            st.success("Deleted from database.")
                            st.rerun()

    with t_add:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0;'>Add New Transaction</h4>", unsafe_allow_html=True)
            with st.form("add_txn_form", clear_on_submit=True):
                na1, na2 = st.columns([3, 2])
                with na1:
                    n_desc = st.text_input("Description / Narration", placeholder="e.g. UPI-SWIGGY-83921, Rent...")
                with na2:
                    n_amt = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")
                na3, na4 = st.columns(2)
                with na3:
                    n_date = st.date_input("Date", value=date.today())
                with na4:
                    n_type = st.selectbox("Type", ["Expense", "Income"])

                if st.form_submit_button("Categorize & Save", type="primary", use_container_width=True):
                    if not n_desc.strip():
                        st.error("Please enter a description.")
                    else:
                        pred = classify_record(n_desc.strip())
                        rec = {
                            "date": str(n_date),
                            "description": n_desc.strip(),
                            "amount": n_amt,
                            "type": n_type,
                            "category": pred["category"] if pred["category"] in SUPPORTED_CATEGORIES else "Other expenses",
                            "confidence": pred["confidence"],
                            "confidence_score": pred["confidence_score"],
                            "prediction_source": pred["source"],
                            "matched_merchant": pred.get("matched_merchant") or "—",
                            "review_status": "User Confirmation Required" if pred["confidence"] == "Low" else "Prediction Accepted"
                        }
                        if db.insert_transaction(rec):
                            st.success(f"✓ Added: {rec['category']} ({rec['confidence']} confidence via {rec['prediction_source']})")
                            st.rerun()
                        else:
                            st.info("Exact duplicate already present in database.")

    with t_import:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 6px 0;'>Import CSV Statement</h4>", unsafe_allow_html=True)
            st.caption("Upload any bank statement CSV. Narration, date, amount, and type are automatically detected.")
            csv_file = st.file_uploader("Choose CSV", type=["csv"], key="import_uploader")
            if csv_file:
                try:
                    raw_df = pd.read_csv(csv_file)
                    st.write(f"Detected **{len(raw_df):,}** rows in file. Preview:")
                    st.dataframe(raw_df.head(5), use_container_width=True)

                    if st.button("Run Hybrid AI Categorization & Import", type="primary"):
                        lookup = {str(c).strip().lower(): c for c in raw_df.columns}
                        def f_col(*names):
                            for n in names:
                                if n in lookup: return lookup[n]
                            return None
                        d_col = f_col("description", "narration", "merchant", "details", "particulars")
                        a_col = f_col("amount", "value", "debit")
                        dt_col = f_col("date", "transaction date", "posted date")
                        t_col = f_col("type", "transaction type")

                        if not d_col:
                            st.error("CSV needs a description column.")
                        else:
                            success_count = 0
                            for _, r in raw_df.iterrows():
                                desc_val = str(r[d_col])
                                amt_val = float(pd.to_numeric(str(r[a_col]).replace(",", ""), errors="coerce") or 0.0) if a_col else 0.0
                                date_val = str(pd.to_datetime(r[dt_col], errors="coerce").date()) if dt_col and pd.notna(r[dt_col]) else str(date.today())
                                type_val = str(r[t_col]).title() if t_col and str(r[t_col]).title() in ["Expense", "Income"] else "Expense"
                                
                                pred = classify_record(desc_val)
                                rec = {
                                    "date": date_val,
                                    "description": desc_val,
                                    "amount": amt_val,
                                    "type": type_val,
                                    "category": pred["category"] if pred["category"] in SUPPORTED_CATEGORIES else "Other expenses",
                                    "confidence": pred["confidence"],
                                    "confidence_score": pred["confidence_score"],
                                    "prediction_source": pred["source"],
                                    "matched_merchant": pred.get("matched_merchant") or "—",
                                    "review_status": "User Confirmation Required" if pred["confidence"] == "Low" else "Prediction Accepted"
                                }
                                if db.insert_transaction(rec):
                                    success_count += 1
                            st.success(f"Imported {success_count:,} new transactions into database.")
                            st.rerun()
                except Exception as ex:
                    st.error(f"Import error: {ex}")

# =========================================================
# 8. PAGE 3: AI CATEGORIZATION (WORKBENCH & LEARNING)
# =========================================================

elif active_page == "AI Categorization":
    render_header("AI Categorization Engine", "Test the 3-tier hybrid pipeline and teach persistent feedback rules.")

    with st.container(border=True):
        st.markdown("<h4 style='margin:0 0 10px 0;'>Single-Transaction AI Classifier</h4>", unsafe_allow_html=True)
        with st.form("wb_predict"):
            q_text = st.text_input("Enter transaction string", placeholder="e.g. UPI-CHAAYOS-83921, DMART GROCERY, COURSERA LEARNING...")
            q_btn = st.form_submit_button("Classify with Hybrid AI", type="primary")

    if q_btn and q_text.strip():
        with st.spinner("Evaluating across tiers..."):
            res = classify_record(q_text.strip())
            st.session_state["wb_res"] = {"text": q_text.strip(), "result": res}

    saved_wb = st.session_state.get("wb_res")
    if saved_wb:
        r = saved_wb["result"]
        with st.container(border=True):
            st.markdown(f"### Classification Result: **{r['category']}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Assigned Category", r["category"])
            m2.metric("Pipeline Source", r["source"])
            m3.metric("Confidence", r["confidence"])
            m4.metric("Margin Score", f"{r['confidence_score']:.4f}")

            st.markdown("---")
            st.write(f"**Cleaned Description:** `{r['cleaned_text']}`")
            st.write(f"**Matched Merchant Rule:** `{r.get('matched_merchant') or 'None'}`")

            # Correction Box
            with st.form("wb_correct"):
                st.write("**Disagreed with this prediction? Teach a persistent rule:**")
                new_cat = st.selectbox("Correct Category", SUPPORTED_CATEGORIES, index=SUPPORTED_CATEGORIES.index(r['category']) if r['category'] in SUPPORTED_CATEGORIES else 0)
                if st.form_submit_button("Save Feedback Rule", type="primary"):
                    teach_correction(saved_wb["text"], r["category"], new_cat)
                    st.success(f"Rule saved! Future matching descriptions will automatically classify as '{new_cat}'.")
                    st.rerun()

# =========================================================
# 9. PAGE 4: REVIEW QUEUE (NEEDS REVIEW ACTION ITEMS)
# =========================================================

elif active_page == "Review Queue":
    render_header("Review Queue", "Transactions requiring confirmation or corrections to ensure accounting accuracy.")

    txns = db.get_all_transactions()
    unresolved = txns[txns["confidence"] == "Low"].copy()

    if unresolved.empty:
        st.success("🎉 All clear! There are zero transactions requiring review.")
    else:
        st.write(f"Found **{len(unresolved)}** items requiring attention:")
        for _, row in unresolved.iterrows():
            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([4, 4, 2])
                with rc1:
                    st.markdown(f"**{row['description']}**")
                    st.caption(f"Date: {row['date'].strftime('%d %b %Y') if pd.notna(row['date']) else '—'} | Amount: {fmt_inr(row['amount'])}")
                with rc2:
                    st.write(f"Predicted Category: **{row['category']}** (Score: {row['confidence_score']:.2f})")
                    c_choice = st.selectbox("Correct to", SUPPORTED_CATEGORIES, index=SUPPORTED_CATEGORIES.index(row["category"]) if row["category"] in SUPPORTED_CATEGORIES else 0, key=f"q_sel_{row['transaction_id']}")
                with rc3:
                    if st.button("Confirm", key=f"q_acc_{row['transaction_id']}", use_container_width=True):
                        db.update_transaction(row["transaction_id"], {
                            "category": c_choice,
                            "confidence": "High",
                            "review_status": "User Confirmed"
                        })
                        teach_correction(row["description"], row["category"], c_choice)
                        st.success("Resolved!")
                        st.rerun()

# =========================================================
# 10. PAGE 5: SPENDING ANALYTICS
# =========================================================

elif active_page == "Spending Analytics":
    render_header("Spending Analytics", "Deep expenditure breakdown, period comparisons, and top category concentrations.")

    txns = db.get_all_transactions()
    expenses = txns[txns["type"] == "Expense"].copy()

    if expenses.empty:
        st.info("No expense data recorded.")
    else:
        a_col1, a_col2 = st.columns([6, 4])
        with a_col1:
            with st.container(border=True):
                st.markdown("<h4 style='margin:0 0 10px 0;'>Category-Wise Expenditure Breakdown</h4>", unsafe_allow_html=True)
                cat_data = expenses.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
                fig_bar = px.bar(cat_data, x="category", y="amount", color="category", color_discrete_map=CATEGORY_PALETTE)
                fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        with a_col2:
            with st.container(border=True):
                st.markdown("<h4 style='margin:0 0 10px 0;'>Top 8 Outflows</h4>", unsafe_allow_html=True)
                top_outflows = expenses.nlargest(8, "amount")[["date", "description", "amount", "category"]].copy()
                top_outflows["Formatted Date"] = top_outflows["date"].dt.strftime("%d %b %Y").fillna("—")
                top_outflows["Formatted Amount"] = top_outflows["amount"].map(fmt_inr)
                st.dataframe(top_outflows[["Formatted Date", "description", "Formatted Amount", "category"]].rename(
                    columns={"Formatted Date": "Date", "description": "Description", "Formatted Amount": "Amount"}
                ), use_container_width=True, hide_index=True)


# =========================================================
# 11. PAGE 6: BUDGETS
# =========================================================

elif active_page == "Budgets":
    render_header("Budget Management", "Set category expenditure ceilings and monitor monthly consumption.")

    budgets = db.get_budgets()
    txns = db.get_all_transactions()
    expenses = txns[txns["type"] == "Expense"].copy()

    b_view, b_edit = st.tabs(["  📊 Budget Gauges  ", "  ⚙️ Configure Limits  "])

    with b_view:
        with st.container(border=True):
            tot_limit = budgets.get("_monthly_total", 61000.0)
            cur_month_str = datetime.now().strftime("%Y-%m")
            m_exp = expenses[expenses["date"].dt.strftime("%Y-%m") == cur_month_str]
            tot_spent = m_exp["amount"].sum()
            cat_spent = m_exp.groupby("category")["amount"].sum().to_dict()

            pct = (tot_spent / tot_limit * 100) if tot_limit else 0
            is_over = tot_spent > tot_limit
            b_color = "#DC2626" if is_over else ("#F2C94C" if pct >= 80 else "#176B45")

            st.markdown(
                f"""
                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-weight:700;">
                        <span>Overall Monthly Budget Limit</span>
                        <span style="color:{b_color};">{fmt_inr(tot_spent)} / {fmt_inr(tot_limit)} ({pct:.1f}%)</span>
                    </div>
                    <div style="background:#E2E8F0; border-radius:10px; height:12px; overflow:hidden;">
                        <div style="width:{min(pct, 100):.1f}%; background:{b_color}; height:100%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Category gauges
            b_cols = st.columns(2)
            for i, cat in enumerate(SUPPORTED_CATEGORIES):
                lim = float(budgets.get(cat, 0.0))
                if lim <= 0: continue
                s_amt = float(cat_spent.get(cat, 0.0))
                c_pct = (s_amt / lim * 100)
                c_col = "#DC2626" if s_amt > lim else ("#F59E0B" if c_pct >= 80 else "#10B981")

                with b_cols[i % 2]:
                    st.markdown(
                        f"""
                        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:12px; border-radius:12px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; font-weight:700; font-size:13px; margin-bottom:4px;">
                                <span>{cat}</span>
                                <span style="color:{c_col};">{fmt_inr(s_amt)} / {fmt_inr(lim)}</span>
                            </div>
                            <div style="background:#F1F5F9; border-radius:6px; height:8px; overflow:hidden;">
                                <div style="width:{min(c_pct, 100):.1f}%; background:{c_col}; height:100%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with b_edit:
        with st.container(border=True):
            st.markdown("<h4 style='margin:0 0 10px 0;'>Adjust Monthly Budget Allowances</h4>", unsafe_allow_html=True)
            with st.form("b_form"):
                new_tot = st.number_input("Overall Monthly Budget (₹)", value=float(budgets.get("_monthly_total", 61000.0)), step=1000.0)
                st.markdown("---")
                new_cats = {}
                ecols = st.columns(2)
                for j, cat in enumerate(SUPPORTED_CATEGORIES):
                    with ecols[j % 2]:
                        new_cats[cat] = st.number_input(f"{cat} (₹)", value=float(budgets.get(cat, 0.0)), step=500.0)

                if st.form_submit_button("Save Budget Caps", type="primary"):
                    to_save = {"_monthly_total": new_tot}
                    to_save.update(new_cats)
                    db.save_budgets(to_save)
                    st.success("Budgets persisted.")
                    st.rerun()

# =========================================================
# 12. PAGE 7: INCOME & CASH FLOW
# =========================================================

elif active_page == "Income & Cash Flow":
    render_header("Income & Cash Flow Summary", "Detailed cash flow accounting, income sources, and liquidity distribution.")

    txns = db.get_all_transactions()
    expenses = txns[txns["type"] == "Expense"].copy()
    incomes = txns[txns["type"] == "Income"].copy()

    tot_inc = incomes["amount"].sum()
    tot_exp = expenses["amount"].sum()
    net = tot_inc - tot_exp

    i1, i2, i3 = st.columns(3)
    i1.metric("Total Inflow Recorded", fmt_inr(tot_inc))
    i2.metric("Total Outflow", fmt_inr(tot_exp))
    i3.metric("Net Cash Flow", f"{'+' if net >= 0 else '-'}{fmt_inr(abs(net))}")

    with st.container(border=True):
        st.markdown("<h4 style='margin:0 0 10px 0;'>Recorded Income Transactions</h4>", unsafe_allow_html=True)
        if incomes.empty:
            st.caption("No income records recorded.")
        else:
            incomes["Formatted Date"] = incomes["date"].dt.strftime("%d %b %Y").fillna("—")
            incomes["Formatted Amount"] = incomes["amount"].map(fmt_inr)
            st.dataframe(incomes[["Formatted Date", "description", "Formatted Amount", "category"]].rename(
                columns={"Formatted Date": "Date", "description": "Source / Narration", "Formatted Amount": "Amount"}
            ), use_container_width=True, hide_index=True)

# =========================================================
# 13. PAGE 8: SETTINGS & DEMO MANAGEMENT
# =========================================================

elif active_page == "Settings":
    render_header("System Settings", "Database maintenance, demo data seeding/reset, and data backup.")

    with st.container(border=True):
        st.markdown("<h4 style='margin:0 0 10px 0;'>Demo Financial Profile Management</h4>", unsafe_allow_html=True)
        st.write("Demonstration baseline: **₹11,50,000 / year (~₹95,833/month salary)** across 3 months of synthetic transactions.")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("Reset / Reload Demo Financial Dataset", type="primary"):
                db.clear_demo_data()
                cnt = db.seed_demo_profile()
                st.success(f"Seeded {cnt} demo transactions.")
                st.rerun()
        with d2:
            if st.button("Remove Demo Records Only"):
                del_cnt = db.clear_demo_data()
                st.info(f"Removed {del_cnt} demo records from database.")
                st.rerun()

    with st.container(border=True):
        st.markdown("<h4 style='margin:0 0 10px 0;'>Database Backup & CSV Export</h4>", unsafe_allow_html=True)
        txns = db.get_all_transactions()
        if not txns.empty:
            csv_data = txns.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export SQLite Data as CSV", data=csv_data, file_name=f"spendulum_backup_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")


# =========================================================
# 14. FOOTER: REACT BITS <LogoLoop /> COMPONENT
# =========================================================

st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

import streamlit.components.v1 as components

components.html(
    """
    <div style="font-family:'Plus Jakarta Sans',-apple-system,sans-serif; width:100%; overflow:hidden;">
        <style>
        .logoloop-container {
            --logoloop-gap: 36px;
            --logoloop-height: 24px;
            position: relative;
            width: 100%;
            overflow: hidden;
            padding: 16px 0;
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(23, 107, 69, 0.12);
            border-radius: 14px;
            box-shadow: 0 4px 16px rgba(16, 75, 52, 0.05);
            mask-image: linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 10%, rgba(0,0,0,1) 90%, rgba(0,0,0,0) 100%);
            -webkit-mask-image: linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 10%, rgba(0,0,0,1) 90%, rgba(0,0,0,0) 100%);
        }
        .logoloop-header {
            text-align: center;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #68756C;
            margin-bottom: 12px;
        }
        .logoloop-track {
            display: flex;
            width: max-content;
            animation: loopScroll 28s linear infinite;
        }
        .logoloop-track:hover {
            animation-play-state: paused;
        }
        @keyframes loopScroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        .logoloop-list {
            display: flex;
            align-items: center;
            gap: var(--logoloop-gap);
            margin: 0;
            padding: 0 18px;
            list-style: none;
        }
        .logoloop-item {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(23, 107, 69, 0.04);
            border: 1px solid rgba(23, 107, 69, 0.12);
            padding: 6px 16px;
            border-radius: 30px;
            font-size: 12px;
            font-weight: 700;
            color: #104B34;
            white-space: nowrap;
            transition: all 0.25s ease;
            cursor: pointer;
            text-decoration: none;
        }
        .logoloop-item:hover {
            transform: scale(1.08);
            background: #176B45;
            color: #F2C94C;
            border-color: #104B34;
            box-shadow: 0 4px 12px rgba(23, 107, 69, 0.25);
        }
        .logoloop-item .logo-icon {
            font-size: 15px;
        }
        </style>

        <div class="logoloop-container" role="region" aria-label="Integrated Banking & FinTech Partners">
            <div class="logoloop-header">Integrated Indian Financial & Payment Gateways</div>
            <div class="logoloop-track">
                <!-- Group 1 -->
                <ul class="logoloop-list">
                    <li class="logoloop-item"><span class="logo-icon">⚡</span> UPI / NPCI</li>
                    <li class="logoloop-item"><span class="logo-icon">🍔</span> Swiggy</li>
                    <li class="logoloop-item"><span class="logo-icon">🛒</span> DMart</li>
                    <li class="logoloop-item"><span class="logo-icon">🚗</span> Uber India</li>
                    <li class="logoloop-item"><span class="logo-icon">🎬</span> Netflix</li>
                    <li class="logoloop-item"><span class="logo-icon">⛽</span> HPCL Fuel</li>
                    <li class="logoloop-item"><span class="logo-icon">💡</span> MSEDCL Bills</li>
                    <li class="logoloop-item"><span class="logo-icon">💊</span> Tata 1mg</li>
                    <li class="logoloop-item"><span class="logo-icon">✈️</span> Cleartrip</li>
                    <li class="logoloop-item"><span class="logo-icon">🎓</span> Coursera</li>
                    <li class="logoloop-item"><span class="logo-icon">🛍️</span> Myntra</li>
                    <li class="logoloop-item"><span class="logo-icon">☕</span> Chaayos</li>
                    <li class="logoloop-item"><span class="logo-icon">🏦</span> HDFC Bank</li>
                    <li class="logoloop-item"><span class="logo-icon">💳</span> PhonePe</li>
                    <li class="logoloop-item"><span class="logo-icon">📱</span> Google Pay</li>
                </ul>
                <!-- Group 2 (seamless infinite scroll duplicate) -->
                <ul class="logoloop-list" aria-hidden="true">
                    <li class="logoloop-item"><span class="logo-icon">⚡</span> UPI / NPCI</li>
                    <li class="logoloop-item"><span class="logo-icon">🍔</span> Swiggy</li>
                    <li class="logoloop-item"><span class="logo-icon">🛒</span> DMart</li>
                    <li class="logoloop-item"><span class="logo-icon">🚗</span> Uber India</li>
                    <li class="logoloop-item"><span class="logo-icon">🎬</span> Netflix</li>
                    <li class="logoloop-item"><span class="logo-icon">⛽</span> HPCL Fuel</li>
                    <li class="logoloop-item"><span class="logo-icon">💡</span> MSEDCL Bills</li>
                    <li class="logoloop-item"><span class="logo-icon">💊</span> Tata 1mg</li>
                    <li class="logoloop-item"><span class="logo-icon">✈️</span> Cleartrip</li>
                    <li class="logoloop-item"><span class="logo-icon">🎓</span> Coursera</li>
                    <li class="logoloop-item"><span class="logo-icon">🛍️</span> Myntra</li>
                    <li class="logoloop-item"><span class="logo-icon">☕</span> Chaayos</li>
                    <li class="logoloop-item"><span class="logo-icon">🏦</span> HDFC Bank</li>
                    <li class="logoloop-item"><span class="logo-icon">💳</span> PhonePe</li>
                    <li class="logoloop-item"><span class="logo-icon">📱</span> Google Pay</li>
                </ul>
            </div>
        </div>
    </div>
    """,
    height=90,
)
