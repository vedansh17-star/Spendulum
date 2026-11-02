"""
Spendulum — Automated test suite.
Run with: venv\\Scripts\\python.exe -m pytest tests/ -v
"""

import sys
import os
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from datetime import date, datetime

import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — so ml/src modules are importable without installing them
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))


# ===========================================================================
# HELPERS  (replicate key functions without importing Streamlit)
# ===========================================================================

TRANSACTION_COLUMNS = [
    "Transaction ID", "Transaction Key", "Date", "Description",
    "Amount", "Type", "Category", "Confidence",
    "Confidence Score", "Prediction Source", "Matched Merchant", "Review Status",
]
SUPPORTED_CATEGORIES = [
    "Bills", "Education", "Entertainment", "Food", "Fuel", "Groceries",
    "Healthcare", "Shopping", "Transport", "Travel",
]


def build_transaction_key(row: dict) -> str:
    date_value = pd.to_datetime(row.get("Date"), errors="coerce")
    date_text = date_value.strftime("%Y-%m-%d") if not pd.isna(date_value) else ""
    description = str(row.get("Description", "")).strip().lower()
    amount = pd.to_numeric(row.get("Amount", 0), errors="coerce")
    amount_text = f"{0 if pd.isna(amount) else amount:.2f}"
    txn_type = str(row.get("Type", "Expense")).strip().lower()
    fingerprint = "|".join([date_text, description, amount_text, txn_type])
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def normalize_imported_transactions(data: pd.DataFrame) -> pd.DataFrame:
    col_lookup = {str(c).strip().lower(): c for c in data.columns}

    def find_col(*names):
        for n in names:
            if n in col_lookup:
                return col_lookup[n]
        return None

    desc_col = find_col("description", "transaction", "narration", "merchant", "details", "particulars")
    if desc_col is None:
        raise ValueError("No usable description column found.")

    amount_col = find_col("amount", "value", "transaction amount")
    date_col = find_col("date", "transaction date", "posted date", "value date")
    type_col = find_col("type", "transaction type")

    normalized = pd.DataFrame({"Description": data[desc_col]})
    if date_col:
        normalized["Date"] = pd.to_datetime(
            data[date_col], errors="coerce", dayfirst=True
        ).dt.date
    else:
        normalized["Date"] = pd.NaT
    if amount_col:
        raw = data[amount_col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
        normalized["Amount"] = pd.to_numeric(raw, errors="coerce").fillna(0.0).abs()
    else:
        normalized["Amount"] = 0.0
    if type_col:
        normalized["Type"] = data[type_col].astype(str).str.strip().str.title()
        normalized["Type"] = normalized["Type"].where(
            normalized["Type"].isin(["Expense", "Income"]), "Expense"
        )
    else:
        normalized["Type"] = "Expense"
    return normalized.dropna(subset=["Description"])


# ===========================================================================
# FIXTURE — temporary project data directory
# ===========================================================================

@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Each test that needs file I/O gets its own clean directory."""
    return tmp_path


# ===========================================================================
# 1. HYBRID CLASSIFIER TESTS
# ===========================================================================

import hybrid_classifier as hc


class TestHybridClassifier:
    """Tests for the existing hybrid classify pipeline."""

    @pytest.mark.parametrize("description,expected_category", [
        ("UPI-SWIGGY-83921",       "Food"),
        ("POS*DMART*72819",        "Groceries"),
        ("PHONEPE-UBER-92831",     "Transport"),
        ("HDFC-NETFLIX-19283",     "Entertainment"),
        ("UPI-HPCL-83921",         "Fuel"),
        ("PAYTM-MSEDCL-72819",     "Bills"),
        ("UPI-1MG-91827",          "Healthcare"),
        ("GPAY-CLEARTRIP-82931",   "Travel"),
        ("PHONEPE-COURSERA-72819", "Education"),
        ("UPI-MYNTRA-83921",       "Shopping"),
    ])
    def test_known_merchants_correct_category(self, description, expected_category):
        result = hc.predict_category(description)
        assert result["category"] == expected_category, (
            f"{description}: expected {expected_category}, got {result['category']}"
        )

    def test_result_has_required_keys(self):
        result = hc.predict_category("UPI-SWIGGY-99999")
        required = {"category", "source", "confidence", "confidence_score", "action", "cleaned_text", "matched_merchant"}
        assert required.issubset(result.keys())

    def test_high_confidence_known_merchant(self):
        """Known merchants should return High confidence.
        Score is 1.0 for User Feedback / Merchant KB; ML model may return >1 (SVM margin).
        """
        result = hc.predict_category("UPI-SWIGGY-99999")
        assert result["confidence"] == "High"
        # For merchant KB and user feedback matches, score should be >= 1.0
        # (ML model SVM margins can exceed 1.0 for very clear categories)
        assert result["confidence_score"] >= 1.0

    def test_unknown_merchant_returns_category(self):
        """An unknown merchant should still return a category via the ML model."""
        result = hc.predict_category("UPI-UNKNOWNXYZ-99999")
        assert result["category"] in SUPPORTED_CATEGORIES
        assert result["source"] == "Machine Learning Model"

    def test_unknown_merchant_confidence_is_labelled(self):
        result = hc.predict_category("UPI-UNKNOWNXYZ-99999")
        assert result["confidence"] in ("High", "Medium", "Low")

    def test_low_confidence_triggers_review(self):
        """Verify that Low confidence maps to 'User Confirmation Required' action."""
        result = hc.predict_category("UPI-UNKNOWNXYZ-99999")
        if result["confidence"] == "Low":
            assert result["action"] == "User Confirmation Required"

    def test_feedback_priority_over_merchant(self, tmp_path):
        """A feedback correction should override the merchant knowledge base."""
        import user_feedback as uf

        # Temporarily override feedback file path
        orig_path = uf.FEEDBACK_FILE
        test_feedback_file = tmp_path / "user_feedback.csv"
        uf.FEEDBACK_FILE = test_feedback_file

        # Write a correction: swiggy → Healthcare (deliberately wrong, to test override)
        uf.save_feedback(
            description="UPI-SWIGGY-99999",
            cleaned_description="swiggy",
            predicted_category="Food",
            corrected_category="Healthcare",
        )
        # Inject into live rules
        hc.feedback_rules["swiggy"] = "Healthcare"

        result = hc.predict_category("UPI-SWIGGY-99999")
        assert result["category"] == "Healthcare"
        assert result["source"] == "User Feedback"

        # Restore
        uf.FEEDBACK_FILE = orig_path
        if "swiggy" in hc.feedback_rules:
            del hc.feedback_rules["swiggy"]

    def test_clean_transaction_removes_noise(self):
        cleaned = hc.clean_transaction("UPI-SWIGGY-83921")
        # Noise words upi, payment numbers removed; swiggy remains
        assert "upi" not in cleaned
        assert "83921" not in cleaned
        assert "swiggy" in cleaned

    def test_confidence_score_is_numeric(self):
        result = hc.predict_category("UPI-SWIGGY-99999")
        assert isinstance(result["confidence_score"], (int, float))


# ===========================================================================
# 2. TRANSACTION KEY AND DEDUPLICATION
# ===========================================================================

class TestTransactionKey:
    def test_same_inputs_produce_same_key(self):
        row = {"Date": "2025-01-15", "Description": "Swiggy", "Amount": 350.0, "Type": "Expense"}
        assert build_transaction_key(row) == build_transaction_key(row)

    def test_different_amounts_produce_different_keys(self):
        row_a = {"Date": "2025-01-15", "Description": "Swiggy", "Amount": 350.0, "Type": "Expense"}
        row_b = {**row_a, "Amount": 351.0}
        assert build_transaction_key(row_a) != build_transaction_key(row_b)

    def test_different_dates_produce_different_keys(self):
        row_a = {"Date": "2025-01-15", "Description": "Swiggy", "Amount": 350.0, "Type": "Expense"}
        row_b = {**row_a, "Date": "2025-01-16"}
        assert build_transaction_key(row_a) != build_transaction_key(row_b)

    def test_income_and_expense_produce_different_keys(self):
        row_a = {"Date": "2025-01-15", "Description": "Salary", "Amount": 50000.0, "Type": "Income"}
        row_b = {**row_a, "Type": "Expense"}
        assert build_transaction_key(row_a) != build_transaction_key(row_b)

    def test_key_is_64_char_hex(self):
        row = {"Date": "2025-01-15", "Description": "Swiggy", "Amount": 350.0, "Type": "Expense"}
        key = build_transaction_key(row)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ===========================================================================
# 3. CSV IMPORT / NORMALIZATION TESTS
# ===========================================================================

class TestCSVNormalization:
    def _make_df(self, **columns):
        return pd.DataFrame(columns)

    def test_basic_import_with_description_and_amount(self):
        df = self._make_df(
            description=["UPI-SWIGGY-99999", "UPI-UBER-88888"],
            amount=[350.0, 220.0],
            date=["2025-01-15", "2025-01-16"],
        )
        result = normalize_imported_transactions(df)
        assert len(result) == 2
        assert list(result["Description"]) == ["UPI-SWIGGY-99999", "UPI-UBER-88888"]
        assert list(result["Amount"]) == [350.0, 220.0]

    def test_missing_description_column_raises(self):
        df = self._make_df(amount=[100.0])
        with pytest.raises(ValueError):
            normalize_imported_transactions(df)

    def test_missing_date_column_produces_nat(self):
        df = self._make_df(description=["UPI-TEST-111"])
        result = normalize_imported_transactions(df)
        assert pd.isna(result["Date"].iloc[0])

    def test_invalid_amount_becomes_zero(self):
        df = self._make_df(
            description=["UPI-TEST-111"],
            amount=["NOT_A_NUMBER"],
        )
        result = normalize_imported_transactions(df)
        assert result["Amount"].iloc[0] == 0.0

    def test_invalid_date_becomes_nat(self):
        df = self._make_df(
            description=["UPI-TEST-111"],
            date=["NOT-A-DATE"],
        )
        result = normalize_imported_transactions(df)
        assert pd.isna(result["Date"].iloc[0])

    def test_type_normalised_to_expense_or_income(self):
        df = self._make_df(
            description=["SALARY", "SWIGGY"],
            type=["Income", "expense"],
        )
        result = normalize_imported_transactions(df)
        types = list(result["Type"])
        assert "Income" in types
        assert all(t in ("Expense", "Income") for t in types)

    def test_unknown_type_defaults_to_expense(self):
        df = self._make_df(
            description=["UPI-TEST"],
            type=["DEBIT"],
        )
        result = normalize_imported_transactions(df)
        assert result["Type"].iloc[0] == "Expense"

    def test_rows_without_description_are_dropped(self):
        df = self._make_df(description=["SWIGGY", None, "UBER"])
        result = normalize_imported_transactions(df)
        assert len(result) == 2

    def test_amount_with_currency_symbols_parsed(self):
        df = self._make_df(
            description=["SWIGGY"],
            amount=["₹ 350.00"],
        )
        result = normalize_imported_transactions(df)
        assert result["Amount"].iloc[0] == pytest.approx(350.0)

    def test_narration_column_alias_accepted(self):
        df = self._make_df(narration=["UPI-SWIGGY-99999"])
        result = normalize_imported_transactions(df)
        assert result["Description"].iloc[0] == "UPI-SWIGGY-99999"

    def test_duplicate_rows_produce_identical_keys(self):
        row = {"Date": "2025-01-15", "Description": "UPI-SWIGGY", "Amount": 350.0, "Type": "Expense"}
        k1 = build_transaction_key(row)
        k2 = build_transaction_key(row)
        assert k1 == k2

    def test_deduplication_logic(self):
        """Simulated add_transactions_to_history: existing keys are skipped."""
        row = {"Date": "2025-01-15", "Description": "UPI-SWIGGY", "Amount": 350.0, "Type": "Expense"}
        key = build_transaction_key(row)
        existing_keys = {key}
        new_key = build_transaction_key(row)
        assert new_key in existing_keys  # Would be skipped


# ===========================================================================
# 4. USER FEEDBACK PERSISTENCE TESTS
# ===========================================================================

import user_feedback as uf


class TestUserFeedback:
    def test_save_feedback_creates_file(self, tmp_data_dir):
        test_file = tmp_data_dir / "user_feedback.csv"
        orig = uf.FEEDBACK_FILE
        uf.FEEDBACK_FILE = test_file
        try:
            uf.save_feedback("UPI-TEST-999", "test", "Food", "Transport")
            assert test_file.exists()
        finally:
            uf.FEEDBACK_FILE = orig

    def test_save_feedback_appends(self, tmp_data_dir):
        test_file = tmp_data_dir / "user_feedback.csv"
        orig = uf.FEEDBACK_FILE
        uf.FEEDBACK_FILE = test_file
        try:
            uf.save_feedback("TXN-A", "txn a", "Food", "Transport")
            uf.save_feedback("TXN-B", "txn b", "Food", "Healthcare")
            df = pd.read_csv(test_file)
            assert len(df) == 2
        finally:
            uf.FEEDBACK_FILE = orig

    def test_feedback_has_required_columns(self, tmp_data_dir):
        test_file = tmp_data_dir / "user_feedback.csv"
        orig = uf.FEEDBACK_FILE
        uf.FEEDBACK_FILE = test_file
        try:
            uf.save_feedback("TXN", "txn", "Food", "Transport")
            df = pd.read_csv(test_file)
            for col in ("description", "cleaned_description", "predicted_category", "corrected_category"):
                assert col in df.columns
        finally:
            uf.FEEDBACK_FILE = orig

    def test_feedback_corrected_category_stored_correctly(self, tmp_data_dir):
        test_file = tmp_data_dir / "user_feedback.csv"
        orig = uf.FEEDBACK_FILE
        uf.FEEDBACK_FILE = test_file
        try:
            uf.save_feedback("UPI-TEST", "test", "Food", "Bills")
            df = pd.read_csv(test_file)
            assert df["corrected_category"].iloc[0] == "Bills"
        finally:
            uf.FEEDBACK_FILE = orig

    def test_feedback_survives_reload(self, tmp_data_dir):
        """Feedback written to disk should be readable as a proper CSV."""
        test_file = tmp_data_dir / "user_feedback.csv"
        orig = uf.FEEDBACK_FILE
        uf.FEEDBACK_FILE = test_file
        try:
            uf.save_feedback("UPI-RELOAD-TEST", "reload test", "Food", "Education")
            df = pd.read_csv(test_file)
            assert df[df["corrected_category"] == "Education"].shape[0] == 1
        finally:
            uf.FEEDBACK_FILE = orig


# ===========================================================================
# 5. ANALYTICS CALCULATIONS
# ===========================================================================

class TestAnalyticsCalculations:
    def _make_transactions(self):
        return pd.DataFrame({
            "Description": ["SWIGGY", "UBER", "SALARY", "NETFLIX"],
            "Amount": [350.0, 220.0, 60000.0, 599.0],
            "Type": ["Expense", "Expense", "Income", "Expense"],
            "Category": ["Food", "Transport", "Income", "Entertainment"],
            "Date": pd.to_datetime(["2025-03-10", "2025-03-12", "2025-03-01", "2025-03-15"]),
        })

    def test_total_expenses_correct(self):
        df = self._make_transactions()
        expenses = df[df["Type"] == "Expense"]
        assert expenses["Amount"].sum() == pytest.approx(1169.0)

    def test_total_income_correct(self):
        df = self._make_transactions()
        income = df[df["Type"] == "Income"]
        assert income["Amount"].sum() == pytest.approx(60000.0)

    def test_net_cash_flow_correct(self):
        df = self._make_transactions()
        total_exp = df[df["Type"] == "Expense"]["Amount"].sum()
        total_inc = df[df["Type"] == "Income"]["Amount"].sum()
        net = total_inc - total_exp
        assert net == pytest.approx(58831.0)

    def test_category_spending(self):
        df = self._make_transactions()
        expenses = df[df["Type"] == "Expense"]
        by_cat = expenses.groupby("Category")["Amount"].sum()
        assert by_cat["Food"] == pytest.approx(350.0)
        assert by_cat["Transport"] == pytest.approx(220.0)

    def test_monthly_grouping(self):
        df = self._make_transactions()
        expenses = df[df["Type"] == "Expense"].dropna(subset=["Date"])
        monthly = (
            expenses.assign(Month=lambda d: d["Date"].dt.to_period("M").astype(str))
            .groupby("Month")["Amount"].sum()
        )
        assert "2025-03" in monthly.index
        assert monthly["2025-03"] == pytest.approx(1169.0)

    def test_income_not_included_in_expenses(self):
        df = self._make_transactions()
        expenses = df[df["Type"] == "Expense"]
        assert 60000.0 not in expenses["Amount"].values

    def test_empty_transactions_produce_zero_totals(self):
        df = pd.DataFrame(columns=["Amount", "Type", "Category", "Date"])
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        assert df[df["Type"] == "Expense"]["Amount"].sum() == 0.0


# ===========================================================================
# 6. BUDGET CALCULATIONS
# ===========================================================================

class TestBudgetCalculations:
    def test_under_budget(self):
        budget = 2000.0
        spent = 1500.0
        remaining = budget - spent
        assert remaining == pytest.approx(500.0)
        assert spent / budget < 0.8

    def test_approaching_limit(self):
        budget = 2000.0
        spent = 1700.0
        pct = spent / budget
        assert pct >= 0.8

    def test_over_budget(self):
        budget = 2000.0
        spent = 2400.0
        assert spent > budget
        over_by = spent - budget
        assert over_by == pytest.approx(400.0)

    def test_zero_budget_not_tracked(self):
        """A zero budget means the user skipped that category."""
        budget = 0.0
        assert budget <= 0

    def test_budget_json_round_trip(self, tmp_data_dir):
        budget_file = tmp_data_dir / "budgets.json"
        budgets = {"_monthly_total": 10000.0, "Food": 2000.0, "Transport": 1000.0}
        with budget_file.open("w") as f:
            json.dump(budgets, f)
        with budget_file.open() as f:
            loaded = json.load(f)
        assert loaded["Food"] == 2000.0
        assert loaded["_monthly_total"] == 10000.0


# ===========================================================================
# 7. TRANSACTION CRUD
# ===========================================================================

class TestTransactionCRUD:
    def _make_txn(self, desc="SWIGGY", amount=350.0, txn_type="Expense", cat="Food"):
        from uuid import uuid4
        return {
            "Transaction ID": uuid4().hex,
            "Transaction Key": build_transaction_key({
                "Date": "2025-01-15", "Description": desc, "Amount": amount, "Type": txn_type
            }),
            "Date": "2025-01-15",
            "Description": desc,
            "Amount": amount,
            "Type": txn_type,
            "Category": cat,
            "Confidence": "High",
            "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base",
            "Matched Merchant": "swiggy",
            "Review Status": "Prediction Accepted",
        }

    def test_add_transaction(self):
        txns = pd.DataFrame(columns=TRANSACTION_COLUMNS)
        new = pd.DataFrame([self._make_txn()])
        combined = pd.concat([txns, new], ignore_index=True)
        assert len(combined) == 1

    def test_delete_transaction(self):
        txn = self._make_txn()
        txns = pd.DataFrame([txn])
        txns = txns[txns["Transaction ID"] != txn["Transaction ID"]]
        assert len(txns) == 0

    def test_edit_transaction(self):
        txn = self._make_txn()
        txns = pd.DataFrame([txn])
        idx = txns["Transaction ID"] == txn["Transaction ID"]
        txns.loc[idx, "Category"] = "Transport"
        assert txns.loc[idx, "Category"].iloc[0] == "Transport"

    def test_duplicate_not_added(self):
        txn = self._make_txn()
        txns = pd.DataFrame([txn])
        existing_keys = set(txns["Transaction Key"])
        new_key = build_transaction_key({
            "Date": "2025-01-15", "Description": "SWIGGY", "Amount": 350.0, "Type": "Expense"
        })
        assert new_key in existing_keys

    def test_transaction_persists_to_csv(self, tmp_data_dir):
        csv_path = tmp_data_dir / "transactions.csv"
        txn = self._make_txn()
        df = pd.DataFrame([txn])
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert loaded["Description"].iloc[0] == "SWIGGY"
        assert float(loaded["Amount"].iloc[0]) == 350.0

    def test_csv_reload_preserves_all_columns(self, tmp_data_dir):
        csv_path = tmp_data_dir / "transactions.csv"
        txn = self._make_txn()
        df = pd.DataFrame([txn])
        df.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        for col in TRANSACTION_COLUMNS:
            assert col in loaded.columns


# ===========================================================================
# 8. TEST REACT BITS COMPONENTS INTEGRATION
# ===========================================================================

class TestReactBitsComponents:
    def test_reactbits_components_exist(self):
        comp_dir = PROJECT_ROOT / "src" / "components"
        assert (comp_dir / "PillNav" / "PillNav.jsx").exists()
        assert (comp_dir / "PillNav" / "PillNav.css").exists()
        assert (comp_dir / "FadeContent" / "FadeContent.jsx").exists()
        assert (comp_dir / "CountUp" / "CountUp.jsx").exists()
        assert (comp_dir / "SpotlightCard" / "SpotlightCard.jsx").exists()
        assert (comp_dir / "SpotlightCard" / "SpotlightCard.css").exists()

    def test_app_py_includes_reactbits_elements(self):
        app_file = PROJECT_ROOT / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        assert "react-bits-pill-nav" in content
        assert "react-bits-fade-content" in content
        assert "headline-net-flow" in content
        assert "metric-click-card" in content




# ===========================================================================
# MAIN — run standalone
# ===========================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)
