import streamlit as st
import pandas as pd
import importlib
import sys
import hashlib
import json
from pathlib import Path
from uuid import uuid4


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Spendulum",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# TRANSACTION / CLASSIFIER HELPERS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ML_SOURCE_DIR = PROJECT_ROOT / "ml" / "src"
TRANSACTION_DATA_FILE = PROJECT_ROOT / "data" / "user_transactions.csv"
INVESTMENT_PROFILE_FILE = PROJECT_ROOT / "data" / "investment_profile.json"
SUPPORTED_CATEGORIES = [
    "Bills", "Education", "Entertainment", "Food", "Fuel", "Groceries",
    "Healthcare", "Shopping", "Transport", "Travel",
]
TRANSACTION_COLUMNS = [
    "Transaction ID", "Transaction Key", "Date", "Description", "Amount", "Type", "Category", "Confidence",
    "Confidence Score", "Prediction Source", "Matched Merchant", "Review Status",
]
DEFAULT_INVESTMENT_PROFILE = {
    "current_savings": 0.0,
    "monthly_income": 0.0,
    "monthly_debt": 0.0,
    "near_term_goal": 0.0,
    "emergency_months": 6,
    "time_horizon": "More than 7 years",
    "risk_comfort": "Moderate",
}


def get_investment_profile():
    """Load the locally saved planning inputs without connecting to any financial account."""
    if "investment_profile" not in st.session_state:
        profile = DEFAULT_INVESTMENT_PROFILE.copy()
        if INVESTMENT_PROFILE_FILE.exists():
            try:
                with INVESTMENT_PROFILE_FILE.open(encoding="utf-8") as profile_file:
                    profile.update(json.load(profile_file))
            except (OSError, json.JSONDecodeError):
                pass
        st.session_state.investment_profile = profile
    return st.session_state.investment_profile.copy()


def save_investment_profile(profile):
    """Persist user-owned planning inputs locally on this computer."""
    INVESTMENT_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INVESTMENT_PROFILE_FILE.open("w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2)
    st.session_state.investment_profile = profile.copy()


def build_transaction_key(transaction):
    """Create a stable key used to prevent importing the same transaction twice."""
    date_value = pd.to_datetime(transaction.get("Date"), errors="coerce")
    date_text = date_value.strftime("%Y-%m-%d") if not pd.isna(date_value) else ""
    description = str(transaction.get("Description", "")).strip().lower()
    amount = pd.to_numeric(transaction.get("Amount", 0), errors="coerce")
    amount_text = f"{0 if pd.isna(amount) else amount:.2f}"
    transaction_type = str(transaction.get("Type", "Expense")).strip().lower()
    fingerprint = "|".join([date_text, description, amount_text, transaction_type])
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def ensure_transaction_schema(transactions):
    """Upgrade loaded records from earlier Spendulum versions without losing data."""
    normalized = transactions.copy()
    for column in TRANSACTION_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized["Transaction ID"] = normalized["Transaction ID"].fillna("")
    missing_ids = normalized["Transaction ID"].eq("")
    normalized.loc[missing_ids, "Transaction ID"] = [uuid4().hex for _ in range(missing_ids.sum())]
    normalized["Transaction Key"] = normalized.apply(build_transaction_key, axis=1)
    return normalized[TRANSACTION_COLUMNS]


def get_transactions():
    """Load the saved transaction history once per app session."""
    if "transactions" not in st.session_state:
        if TRANSACTION_DATA_FILE.exists():
            st.session_state.transactions = ensure_transaction_schema(pd.read_csv(TRANSACTION_DATA_FILE))
        else:
            st.session_state.transactions = pd.DataFrame(columns=TRANSACTION_COLUMNS)
    else:
        st.session_state.transactions = ensure_transaction_schema(st.session_state.transactions)
    return st.session_state.transactions.copy()


def persist_transactions(transactions=None):
    """Persist the current transaction history for the next Spendulum session."""
    records = ensure_transaction_schema(
        st.session_state.transactions if transactions is None else transactions
    )
    TRANSACTION_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(TRANSACTION_DATA_FILE, index=False)
    st.session_state.transactions = records


def add_transactions_to_history(transactions):
    """Append only new records, save them locally, and report how many were accepted."""
    new_transactions = ensure_transaction_schema(transactions)
    existing_transactions = get_transactions()
    existing_keys = set(existing_transactions["Transaction Key"].dropna())
    new_transactions = new_transactions[
        ~new_transactions["Transaction Key"].isin(existing_keys)
    ].drop_duplicates(subset=["Transaction Key"], keep="first")
    if new_transactions.empty:
        return 0

    combined = pd.concat([existing_transactions, new_transactions], ignore_index=True)
    persist_transactions(combined)
    return len(new_transactions)


@st.cache_resource(show_spinner=False)
def load_hybrid_classifier():
    """Load Phase 1's existing hybrid classifier only when it is needed."""
    if str(ML_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(ML_SOURCE_DIR))

    module = importlib.import_module("hybrid_classifier")
    return module


@st.cache_resource(show_spinner=False)
def load_feedback_writer():
    """Load the existing Phase 1 feedback writer when a correction is saved."""
    if str(ML_SOURCE_DIR) not in sys.path:
        sys.path.insert(0, str(ML_SOURCE_DIR))

    return importlib.import_module("user_feedback")


def classify_transactions(transactions):
    """Add Phase 1 hybrid-classifier results to a normalized transaction table."""
    predict_category = load_hybrid_classifier().predict_category
    classified = transactions.copy()
    if "Transaction ID" not in classified.columns:
        classified.insert(0, "Transaction ID", [uuid4().hex for _ in range(len(classified))])
    results = classified["Description"].fillna("").astype(str).map(predict_category)

    classified["Category"] = results.map(lambda result: result["category"])
    classified["Confidence"] = results.map(lambda result: result["confidence"])
    classified["Confidence Score"] = results.map(lambda result: result["confidence_score"])
    classified["Prediction Source"] = results.map(lambda result: result["source"])
    classified["Matched Merchant"] = results.map(
        lambda result: result["matched_merchant"] or "—"
    )
    classified["Review Status"] = results.map(lambda result: result["action"])
    return classified


def learn_from_correction(description, predicted_category, corrected_category):
    """Save a correction and immediately add it to the live classifier rules."""
    classifier = load_hybrid_classifier()
    cleaned_description = classifier.clean_transaction(description)
    feedback_writer = load_feedback_writer()
    feedback_writer.save_feedback(
        description=description,
        cleaned_description=cleaned_description,
        predicted_category=predicted_category,
        corrected_category=corrected_category,
    )

    # The classifier normally loads feedback on startup. Update the live rule map too,
    # so a matching transaction added in this session learns right away.
    classifier.feedback_rules[cleaned_description] = corrected_category


def save_user_correction(transaction_id, description, predicted_category, corrected_category):
    """Save a transaction-table correction and update its displayed result."""
    learn_from_correction(description, predicted_category, corrected_category)

    transaction_rows = st.session_state.transactions["Transaction ID"] == transaction_id
    st.session_state.transactions.loc[transaction_rows, "Category"] = corrected_category
    st.session_state.transactions.loc[transaction_rows, "Confidence"] = "High"
    st.session_state.transactions.loc[transaction_rows, "Confidence Score"] = 1.0
    st.session_state.transactions.loc[transaction_rows, "Prediction Source"] = "User Feedback"
    st.session_state.transactions.loc[transaction_rows, "Matched Merchant"] = "—"
    st.session_state.transactions.loc[transaction_rows, "Review Status"] = "User correction saved"
    persist_transactions()


def normalize_imported_transactions(uploaded_data):
    """Accept common CSV headings and turn them into Spendulum transaction fields."""
    column_lookup = {str(column).strip().lower(): column for column in uploaded_data.columns}

    def find_column(*names):
        for name in names:
            if name in column_lookup:
                return column_lookup[name]
        return None

    description_column = find_column("description", "transaction", "narration", "merchant", "details")
    if description_column is None:
        raise ValueError(
            "Your CSV needs a description column (for example: description, transaction, narration, merchant, or details)."
        )

    amount_column = find_column("amount", "value", "transaction amount")
    date_column = find_column("date", "transaction date", "posted date")
    type_column = find_column("type", "transaction type", "debit credit")

    normalized = pd.DataFrame({"Description": uploaded_data[description_column]})
    normalized["Date"] = (
        pd.to_datetime(uploaded_data[date_column], errors="coerce").dt.date
        if date_column else pd.NaT
    )
    normalized["Amount"] = (
        pd.to_numeric(
            uploaded_data[amount_column].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)
        if amount_column else 0.0
    )
    normalized["Type"] = uploaded_data[type_column].astype(str).str.title() if type_column else "Expense"
    normalized["Type"] = normalized["Type"].where(
        normalized["Type"].isin(["Expense", "Income"]), "Expense"
    )
    return normalized.dropna(subset=["Description"])


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
        background: #0b0f19;
    }


    /* Remove top padding */

    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }


    /* Main title */

    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }


    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 35px;
    }


    /* Metric cards */

    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 22px;
        height: 140px;
    }


    .metric-label {
        color: #9ca3af;
        font-size: 14px;
    }


    .metric-value {
        font-size: 30px;
        font-weight: 700;
        margin-top: 8px;
    }


    .metric-icon {
        font-size: 24px;
    }


    /* Section cards */

    .section-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
    }


    /* AI card */

    .ai-card {
        background: linear-gradient(
            135deg,
            #171c2c,
            #111827
        );

        border: 1px solid #293249;

        border-radius: 18px;

        padding: 25px;

        margin-top: 20px;
    }


    .ai-title {
        font-size: 22px;
        font-weight: 700;
    }


    .ai-text {
        color: #9ca3af;
        margin-top: 8px;
    }


    /* Hide Streamlit menu */

    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="font-size:30px;">
        💸 Spendulum
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "AI-Powered Personal Finance"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "💳 Transactions",
            "🤖 AI Categorization",
            "📊 Analytics",
            "💡 Savings & Investing",
            "⚙️ Settings"
        ]
    )

    st.divider()

    st.caption(
        "Spend smarter. Understand better."
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">Good afternoon 👋</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Here is your financial overview.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    transactions = get_transactions()
    transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce").fillna(0.0)
    expenses = transactions[transactions["Type"] == "Expense"].copy()
    spending_data = (
        expenses.groupby("Category", as_index=False)["Amount"].sum()
        .sort_values("Amount", ascending=False)
    )
    total_spending = expenses["Amount"].sum()
    review_count = (transactions["Confidence"] == "Low").sum()
    dated_expenses = pd.to_datetime(expenses["Date"], errors="coerce").dropna()
    observed_months = max(dated_expenses.dt.to_period("M").nunique(), 1)
    monthly_average = total_spending / observed_months
    top_category = spending_data.iloc[0]["Category"] if not spending_data.empty else "—"

    col1, col2, col3, col4 = st.columns(4)

    def render_metric_card(icon, label, value):
        """Render a dashboard metric as HTML, rather than Markdown text."""
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
        )

    with col1:
        render_metric_card("💰", "Total Spending", f"₹{total_spending:,.0f}")
    with col2:
        render_metric_card("📈", "Monthly Average", f"₹{monthly_average:,.0f}")
    with col3:
        render_metric_card("💳", "Transactions", f"{len(transactions):,}")
    with col4:
        render_metric_card("🤖", "Needs Review", f"{review_count:,}")

    st.markdown(
        """
        <div class="section-card">
        <h2>📊 Spending Overview</h2>
        <p style="color:#9ca3af;">Your live expense totals by AI category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if spending_data.empty:
        st.info("Add or import transactions to see your live spending overview.")
    else:
        st.bar_chart(spending_data.set_index("Category"))

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card"><h3>💸 Category Breakdown</h3></div>', unsafe_allow_html=True)
        if spending_data.empty:
            st.caption("No expense categories yet.")
        else:
            category_table = spending_data.copy()
            category_table["Amount"] = category_table["Amount"].map(lambda amount: f"₹{amount:,.2f}")
            st.dataframe(category_table, use_container_width=True, hide_index=True)

    with right:
        if spending_data.empty:
            insight = "Add transactions to receive a live spending insight from Spendulum."
        else:
            top_amount = spending_data.iloc[0]["Amount"]
            share = (top_amount / total_spending * 100) if total_spending else 0
            insight = (
                f"{top_category} is your highest spending category at ₹{top_amount:,.0f}, "
                f"which is {share:.0f}% of your recorded expenses."
            )
        st.html(
            f"""
            <div class="ai-card">
                <div class="ai-title">🤖 Spendulum AI Insight</div>
                <div class="ai-text">{insight}</div>
            </div>
            """
        )

        st.markdown('<div class="section-card"><h3>🕒 Recent Transactions</h3></div>', unsafe_allow_html=True)
        if transactions.empty:
            st.caption("Your recent transactions will appear here.")
        else:
            recent_transactions = transactions.tail(5).iloc[::-1][["Date", "Description", "Amount", "Category"]].copy()
            recent_transactions["Amount"] = recent_transactions["Amount"].map(lambda amount: f"₹{amount:,.2f}")
            st.dataframe(recent_transactions, use_container_width=True, hide_index=True)


# =========================================================
# TRANSACTIONS PAGE
# =========================================================

elif page == "💳 Transactions":

    st.markdown('<div class="main-title">💳 Transactions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Add or import activity and let your Phase 1 hybrid AI categorize it.</div>',
        unsafe_allow_html=True,
    )

    current_transactions = get_transactions()
    # Preserve transactions that were added before persistent storage was introduced.
    if not current_transactions.empty and not TRANSACTION_DATA_FILE.exists():
        persist_transactions(current_transactions)

    entry_tab, import_tab = st.tabs(["➕ Add transaction", "📂 Import CSV"])

    with entry_tab:
        st.markdown('<div class="section-card"><h3>Add a transaction</h3></div>', unsafe_allow_html=True)
        with st.form("add_transaction", clear_on_submit=True):
            first_column, second_column = st.columns([2, 1])
            with first_column:
                description = st.text_input(
                    "Description", placeholder="e.g. UPI-SWIGGY-83921"
                )
            with second_column:
                amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")

            date_column, type_column = st.columns(2)
            with date_column:
                transaction_date = st.date_input("Date")
            with type_column:
                transaction_type = st.selectbox("Type", ["Expense", "Income"])

            submitted = st.form_submit_button("Categorize and add", use_container_width=True)

        if submitted:
            if not description.strip():
                st.error("Please enter a transaction description.")
            else:
                try:
                    new_transaction = pd.DataFrame(
                        [{
                            "Date": transaction_date,
                            "Description": description.strip(),
                            "Amount": amount,
                            "Type": transaction_type,
                        }]
                    )
                    classified = classify_transactions(new_transaction)
                    added_count = add_transactions_to_history(classified)
                    result = classified.iloc[0]
                    if added_count:
                        st.success(
                            f"Added as {result['Category']} · {result['Confidence']} confidence · {result['Prediction Source']}"
                        )
                    else:
                        st.info("This exact transaction is already in your saved history.")
                except Exception as error:
                    st.error(f"Spendulum could not load the Phase 1 classifier: {error}")

    with import_tab:
        st.markdown('<div class="section-card"><h3>Import transactions</h3></div>', unsafe_allow_html=True)
        st.caption("Use a CSV with a description column. Date, amount, and type are optional.")
        uploaded_file = st.file_uploader("Transaction CSV", type=["csv"], key="transaction_csv")

        if uploaded_file is not None:
            try:
                preview = pd.read_csv(uploaded_file)
                st.caption(f"{len(preview):,} rows ready to import")
                st.dataframe(preview.head(10), use_container_width=True, hide_index=True)

                if st.button("Categorize and import", type="primary", use_container_width=True):
                    normalized = normalize_imported_transactions(preview)
                    if normalized.empty:
                        st.warning("There are no transactions with a usable description to import.")
                    else:
                        with st.spinner("Categorizing transactions with Spendulum AI..."):
                            classified = classify_transactions(normalized)
                        added_count = add_transactions_to_history(classified)
                        skipped_count = len(classified) - added_count
                        if added_count:
                            message = f"Saved and categorized {added_count:,} new transactions."
                            if skipped_count:
                                message += f" Skipped {skipped_count:,} exact duplicate(s)."
                            st.success(message)
                        else:
                            st.info("Every row in this CSV is already in your saved history.")
                        st.rerun()
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                st.error(f"This CSV could not be read: {error}")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Import failed: {error}")

    transactions = st.session_state.transactions.copy()
    st.markdown('<div class="section-card"><h3>Your transaction history</h3></div>', unsafe_allow_html=True)

    if transactions.empty:
        st.info("No transactions yet. Add one manually or import a CSV to start your history.")
    else:
        if "Transaction ID" not in transactions.columns:
            transactions.insert(0, "Transaction ID", [uuid4().hex for _ in range(len(transactions))])
            st.session_state.transactions = transactions.copy()

        transactions["Amount"] = pd.to_numeric(transactions["Amount"], errors="coerce").fillna(0.0)
        filters, totals = st.columns([3, 1])
        with filters:
            category_options = sorted(transactions["Category"].dropna().unique().tolist())
            selected_categories = st.multiselect("Filter by category", category_options)
        with totals:
            expense_total = transactions.loc[transactions["Type"] == "Expense", "Amount"].sum()
            st.metric("Imported spending", f"₹{expense_total:,.2f}")

        if selected_categories:
            transactions = transactions[transactions["Category"].isin(selected_categories)]

        low_confidence_count = (transactions["Confidence"] == "Low").sum()
        if low_confidence_count:
            st.warning(f"{low_confidence_count} transaction(s) need your confirmation before you rely on their category.")

            with st.expander("✍️ Review low-confidence predictions", expanded=True):
                st.caption("Choose the correct category. Spendulum will save the correction and learn it immediately.")
                low_confidence_rows = transactions[transactions["Confidence"] == "Low"]

                for _, row in low_confidence_rows.iterrows():
                    current_category = row["Category"]
                    default_index = (
                        SUPPORTED_CATEGORIES.index(current_category)
                        if current_category in SUPPORTED_CATEGORIES else 0
                    )
                    transaction_label = (
                        f"{row['Description']} · ₹{row['Amount']:,.2f} "
                        f"(currently: {current_category})"
                    )

                    with st.form(f"correction_{row['Transaction ID']}"):
                        corrected_category = st.selectbox(
                            transaction_label,
                            SUPPORTED_CATEGORIES,
                            index=default_index,
                            key=f"corrected_category_{row['Transaction ID']}",
                        )
                        save_correction = st.form_submit_button("Save correction")

                    if save_correction:
                        try:
                            save_user_correction(
                                transaction_id=row["Transaction ID"],
                                description=row["Description"],
                                predicted_category=current_category,
                                corrected_category=corrected_category,
                            )
                            st.success(
                                f"Saved: {row['Description']} → {corrected_category}. "
                                "Future matching transactions will use this feedback."
                            )
                            st.rerun()
                        except Exception as error:
                            st.error(f"Correction could not be saved: {error}")

        display_transactions = transactions.copy()
        display_transactions["Amount"] = display_transactions["Amount"].map(lambda value: f"₹{value:,.2f}")
        st.dataframe(
            display_transactions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Transaction ID": None,
                "Transaction Key": None,
                "Confidence Score": st.column_config.NumberColumn(format="%.4f"),
                "Date": st.column_config.DateColumn(format="DD MMM YYYY"),
            },
        )


# =========================================================
# AI CATEGORIZATION PAGE
# =========================================================

elif page == "🤖 AI Categorization":

    st.markdown('<div class="main-title">🤖 AI Categorization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Test a single transaction and inspect exactly how Spendulum classified it.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-card"><h3>Classify a transaction</h3></div>', unsafe_allow_html=True)
    with st.form("single_transaction_prediction"):
        transaction_description = st.text_input(
            "Transaction description",
            placeholder="e.g. UPI-SWIGGY-83921",
        )
        predict_single_transaction = st.form_submit_button(
            "Run Spendulum AI", type="primary", use_container_width=True
        )

    if predict_single_transaction:
        if not transaction_description.strip():
            st.error("Enter a transaction description first.")
        else:
            try:
                with st.spinner("Checking feedback, merchant rules, and the ML model..."):
                    prediction = load_hybrid_classifier().predict_category(
                        transaction_description.strip()
                    )
                st.session_state.ai_prediction = {
                    "description": transaction_description.strip(),
                    "result": prediction,
                }
            except Exception as error:
                st.error(f"Spendulum could not run the classifier: {error}")

    saved_prediction = st.session_state.get("ai_prediction")
    if saved_prediction:
        description = saved_prediction["description"]
        result = saved_prediction["result"]
        st.markdown('<div class="section-card"><h3>Prediction result</h3></div>', unsafe_allow_html=True)

        category_column, confidence_column, source_column, score_column = st.columns(4)
        category_column.metric("Category", result["category"])
        confidence_column.metric("Confidence", result["confidence"])
        source_column.metric("Source", result["source"])
        score_column.metric("Confidence score", f"{result['confidence_score']:.4f}")

        if result["confidence"] == "Low":
            st.warning("This prediction is uncertain. Confirm or correct the category below.")
        else:
            st.success("This prediction has sufficient confidence.")

        with st.expander("See classifier details"):
            st.write(f"**Original description:** {description}")
            st.write(f"**Cleaned description:** {result['cleaned_text'] or 'No remaining keywords'}")
            st.write(f"**Matched merchant:** {result['matched_merchant'] or 'No merchant rule matched'}")
            st.write(f"**Decision:** {result['action']}")

        with st.expander("✍️ Correct this prediction", expanded=result["confidence"] == "Low"):
            st.caption("A correction is saved to the feedback-learning pipeline and is used immediately.")
            default_index = (
                SUPPORTED_CATEGORIES.index(result["category"])
                if result["category"] in SUPPORTED_CATEGORIES else 0
            )
            with st.form("single_prediction_correction"):
                corrected_category = st.selectbox(
                    "Correct category",
                    SUPPORTED_CATEGORIES,
                    index=default_index,
                )
                save_ai_correction = st.form_submit_button("Save correction")

            if save_ai_correction:
                if corrected_category == result["category"]:
                    st.info("The selected category matches the current prediction, so no correction was saved.")
                else:
                    try:
                        learn_from_correction(
                            description=description,
                            predicted_category=result["category"],
                            corrected_category=corrected_category,
                        )
                        st.session_state.ai_prediction["result"] = {
                            **result,
                            "category": corrected_category,
                            "source": "User Feedback",
                            "matched_merchant": None,
                            "confidence": "High",
                            "confidence_score": 1.0,
                            "action": "User correction saved",
                        }
                        st.success(
                            f"Learned: {description} → {corrected_category}. "
                            "Matching future descriptions will use this feedback."
                        )
                        st.rerun()
                    except Exception as error:
                        st.error(f"Correction could not be saved: {error}")


# =========================================================
# ANALYTICS PAGE
# =========================================================

elif page == "📊 Analytics":

    st.markdown('<div class="main-title">📊 Financial Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Explore your saved spending patterns and category trends.</div>',
        unsafe_allow_html=True,
    )

    analytics_transactions = get_transactions()
    analytics_transactions["Amount"] = pd.to_numeric(
        analytics_transactions["Amount"], errors="coerce"
    ).fillna(0.0)
    analytics_transactions["Date"] = pd.to_datetime(
        analytics_transactions["Date"], errors="coerce"
    )
    expenses = analytics_transactions[
        analytics_transactions["Type"] == "Expense"
    ].copy()

    if expenses.empty:
        st.info("Add or import transactions to unlock your spending analytics.")
    else:
        filter_column, category_column = st.columns(2)
        dated_expenses = expenses.dropna(subset=["Date"])

        with filter_column:
            if dated_expenses.empty:
                selected_dates = None
                st.caption("Date filtering is unavailable because these records have no valid dates.")
            else:
                earliest_date = dated_expenses["Date"].min().date()
                latest_date = dated_expenses["Date"].max().date()
                selected_dates = st.date_input(
                    "Date range",
                    value=(earliest_date, latest_date),
                    min_value=earliest_date,
                    max_value=latest_date,
                )

        with category_column:
            category_options = sorted(expenses["Category"].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "Categories",
                category_options,
                placeholder="All categories",
            )

        filtered_expenses = expenses.copy()
        if selected_dates and len(selected_dates) == 2:
            start_date, end_date = pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])
            filtered_expenses = filtered_expenses[
                filtered_expenses["Date"].between(start_date, end_date, inclusive="both")
            ]
        if selected_categories:
            filtered_expenses = filtered_expenses[
                filtered_expenses["Category"].isin(selected_categories)
            ]

        if filtered_expenses.empty:
            st.warning("No expenses match the selected filters.")
        else:
            category_spending = (
                filtered_expenses.groupby("Category", as_index=False)["Amount"].sum()
                .sort_values("Amount", ascending=False)
            )
            monthly_spending = (
                filtered_expenses.dropna(subset=["Date"])
                .assign(Month=lambda frame: frame["Date"].dt.to_period("M").astype(str))
                .groupby("Month", as_index=False)["Amount"].sum()
                .sort_values("Month")
            )
            total_spending = filtered_expenses["Amount"].sum()
            average_transaction = filtered_expenses["Amount"].mean()
            top_category = category_spending.iloc[0]
            largest_month = (
                monthly_spending.loc[monthly_spending["Amount"].idxmax()]
                if not monthly_spending.empty else None
            )

            metric_one, metric_two, metric_three, metric_four = st.columns(4)
            metric_one.metric("Filtered spending", f"₹{total_spending:,.0f}")
            metric_two.metric("Expense transactions", f"{len(filtered_expenses):,}")
            metric_three.metric("Average transaction", f"₹{average_transaction:,.0f}")
            metric_four.metric("Top category", top_category["Category"])

            trend_column, category_chart_column = st.columns(2)
            with trend_column:
                st.markdown('<div class="section-card"><h3>📈 Monthly spending trend</h3></div>', unsafe_allow_html=True)
                if monthly_spending.empty:
                    st.caption("Import dates to see the monthly trend.")
                else:
                    st.line_chart(monthly_spending.set_index("Month"))

            with category_chart_column:
                st.markdown('<div class="section-card"><h3>🥧 Category comparison</h3></div>', unsafe_allow_html=True)
                st.bar_chart(category_spending.set_index("Category"))

            table_column, insight_column = st.columns(2)
            with table_column:
                st.markdown('<div class="section-card"><h3>💸 Category totals</h3></div>', unsafe_allow_html=True)
                category_table = category_spending.copy()
                category_table["Amount"] = category_table["Amount"].map(lambda amount: f"₹{amount:,.2f}")
                category_table["Share"] = (
                    category_spending["Amount"] / total_spending * 100
                ).map(lambda share: f"{share:.1f}%")
                st.dataframe(category_table, use_container_width=True, hide_index=True)

            with insight_column:
                largest_month_text = (
                    f" Your highest month was {largest_month['Month']} at ₹{largest_month['Amount']:,.0f}."
                    if largest_month is not None else ""
                )
                top_share = top_category["Amount"] / total_spending * 100
                st.html(
                    f"""
                    <div class="ai-card">
                        <div class="ai-title">🤖 Spending signal</div>
                        <div class="ai-text">
                            {top_category['Category']} accounts for {top_share:.0f}% of the selected spending
                            (₹{top_category['Amount']:,.0f}).{largest_month_text}
                        </div>
                    </div>
                    """
                )


# =========================================================
# SAVINGS & INVESTMENT READINESS PAGE
# =========================================================

elif page == "💡 Savings & Investing":

    st.markdown('<div class="main-title">💡 Savings & Investing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Understand your investment readiness before putting money at risk.</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "Educational planning tool only — it does not provide investment advice, recommend securities, "
        "or guarantee returns."
    )

    profile = get_investment_profile()
    st.markdown('<div class="section-card"><h3>Your planning inputs</h3></div>', unsafe_allow_html=True)
    with st.form("investment_profile_form"):
        first_column, second_column, third_column = st.columns(3)
        with first_column:
            current_savings = st.number_input(
                "Current savings (₹)", min_value=0.0,
                value=float(profile["current_savings"]), step=1000.0,
            )
            monthly_income = st.number_input(
                "Monthly take-home income (₹)", min_value=0.0,
                value=float(profile["monthly_income"]), step=1000.0,
            )
        with second_column:
            monthly_debt = st.number_input(
                "Monthly debt / EMI payments (₹)", min_value=0.0,
                value=float(profile["monthly_debt"]), step=500.0,
            )
            near_term_goal = st.number_input(
                "Savings reserved for near-term goals (₹)", min_value=0.0,
                value=float(profile["near_term_goal"]), step=1000.0,
            )
        with third_column:
            emergency_months = st.slider(
                "Emergency reserve target (months of expenses)",
                min_value=3, max_value=12, value=int(profile["emergency_months"]),
            )
            time_horizon = st.selectbox(
                "Investment time horizon",
                ["Less than 3 years", "3 to 7 years", "More than 7 years"],
                index=["Less than 3 years", "3 to 7 years", "More than 7 years"].index(profile["time_horizon"]),
            )
            risk_comfort = st.selectbox(
                "Comfort with value fluctuations",
                ["Conservative", "Moderate", "Growth-oriented"],
                index=["Conservative", "Moderate", "Growth-oriented"].index(profile["risk_comfort"]),
            )

        save_profile = st.form_submit_button("Save planning inputs", use_container_width=True)

    if save_profile:
        profile = {
            "current_savings": current_savings,
            "monthly_income": monthly_income,
            "monthly_debt": monthly_debt,
            "near_term_goal": near_term_goal,
            "emergency_months": emergency_months,
            "time_horizon": time_horizon,
            "risk_comfort": risk_comfort,
        }
        save_investment_profile(profile)
        st.success("Your local planning inputs were saved.")

    planning_transactions = get_transactions()
    planning_transactions["Amount"] = pd.to_numeric(
        planning_transactions["Amount"], errors="coerce"
    ).fillna(0.0)
    planning_transactions["Date"] = pd.to_datetime(
        planning_transactions["Date"], errors="coerce"
    )
    expense_history = planning_transactions[
        planning_transactions["Type"] == "Expense"
    ].dropna(subset=["Date"])
    monthly_expenses = (
        expense_history
        .assign(Month=lambda frame: frame["Date"].dt.to_period("M"))
        .groupby("Month")["Amount"].sum()
    )
    monthly_expense_baseline = monthly_expenses.mean() if not monthly_expenses.empty else 0.0
    emergency_target = monthly_expense_baseline * profile["emergency_months"]
    emergency_gap = max(0.0, emergency_target - profile["current_savings"])
    protected_savings = emergency_target + profile["near_term_goal"]
    investable_lump_sum = max(0.0, profile["current_savings"] - protected_savings)
    monthly_investment_capacity = max(
        0.0,
        profile["monthly_income"] - monthly_expense_baseline - profile["monthly_debt"],
    )

    st.markdown('<div class="section-card"><h3>Investment readiness snapshot</h3></div>', unsafe_allow_html=True)
    metric_one, metric_two, metric_three, metric_four = st.columns(4)
    metric_one.metric("Average monthly expenses", f"₹{monthly_expense_baseline:,.0f}")
    metric_two.metric("Emergency reserve target", f"₹{emergency_target:,.0f}")
    metric_three.metric("Estimated investable savings", f"₹{investable_lump_sum:,.0f}")
    metric_four.metric("Monthly investment capacity", f"₹{monthly_investment_capacity:,.0f}")
    st.caption(
        f"Expense baseline is calculated from {len(expense_history):,} dated expenses across "
        f"{len(monthly_expenses):,} month(s). Import personal transactions for a personal estimate."
    )

    emergency_progress = (
        min(profile["current_savings"] / emergency_target, 1.0)
        if emergency_target else 0.0
    )
    st.caption(
        f"Emergency reserve progress: ₹{profile['current_savings']:,.0f} of ₹{emergency_target:,.0f}"
    )
    st.progress(emergency_progress)

    if monthly_expense_baseline == 0:
        readiness_message = "Import dated expense transactions first so Spendulum can estimate your monthly expense baseline."
    elif emergency_gap > 0:
        readiness_message = (
            f"Build your emergency reserve first. You are ₹{emergency_gap:,.0f} below your selected "
            f"{profile['emergency_months']}-month reserve target."
        )
    elif monthly_investment_capacity <= 0:
        readiness_message = (
            "Your current income minus expenses and debt leaves no monthly investment capacity. "
            "Focus on cash flow or debt before taking additional investment risk."
        )
    else:
        readiness_message = (
            "Your selected emergency reserve is funded and your budget shows positive monthly capacity. "
            "Use your goals, horizon, and risk comfort to research a diversified approach."
        )
    st.info(readiness_message)

    if profile["time_horizon"] == "Less than 3 years":
        horizon_guidance = "For a short horizon, prioritise liquidity and capital preservation over volatile investments."
    elif profile["time_horizon"] == "3 to 7 years":
        horizon_guidance = "For a medium horizon, consider how a diversified mix could balance growth potential and volatility."
    else:
        horizon_guidance = "For a longer horizon, understand how diversification and regular investing can affect volatility and compounding."

    risk_guidance = {
        "Conservative": "Your selected comfort level suggests avoiding a plan that depends on large short-term price swings.",
        "Moderate": "Your selected comfort level suggests balancing growth goals with your ability to tolerate drawdowns.",
        "Growth-oriented": "Higher growth expectations can involve larger losses and volatility; only take risk you can sustain.",
    }[profile["risk_comfort"]]

    st.html(
        f"""
        <div class="ai-card">
            <div class="ai-title">🧭 Planning guidance</div>
            <div class="ai-text">
                {horizon_guidance}<br><br>
                {risk_guidance}<br><br>
                Before investing, consider emergency savings, insurance, debt, goals, time horizon, and risk tolerance.
            </div>
        </div>
        """
    )
    st.markdown(
        "Learn more through [SEBI’s investor education resources](https://investor.sebi.gov.in/why_one_should_invest.html) "
        "and review the [SEBI Riskometer guidance](https://investor.sebi.gov.in/hindi/riskometer.html)."
    )


# =========================================================
# SETTINGS PAGE
# =========================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.write(
        "Spendulum configuration and user preferences."
    )
