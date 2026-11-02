# Spendulum

AI-powered personal finance and expense management application for everyday Indian financial transactions.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vedansh17-star/Spendulum.git
cd Spendulum
```

### 2. Set Up Virtual Environment & Dependencies
**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## Features

| Feature | Status |
|---|---|
| Professional green-yellow-white financial UI | ✅ Complete |
| Dashboard with real-data metrics | ✅ Complete |
| Transaction add / edit / delete | ✅ Complete |
| CSV import with validation and preview | ✅ Complete |
| Hybrid AI categorization (Feedback → Merchant KB → SVM) | ✅ Complete |
| Category correction and feedback learning | ✅ Complete |
| Financial analytics with date/category filters | ✅ Complete |
| Monthly budget tracking with progress bars | ✅ Complete |
| Data-supported spending insights | ✅ Complete |
| Settings and CSV export | ✅ Complete |
| Automated test suite (58 tests) | ✅ Complete |

---

## Navigation

| Page | Purpose |
|---|---|
| **Overview** | Dashboard — expenses, income, net cash flow, trends, insights |
| **Transactions** | Add, import CSV, search/filter, edit, delete, review low-confidence |
| **AI Categorization** | Test the classifier, view details, save corrections |
| **Analytics** | Category breakdown, income vs expenses, monthly trends |
| **Budgets** | Set monthly limits per category, track progress |
| **Settings** | Currency, date format, export, data paths |

---

## File Layout

```
Spendulum/
├── app.py                          # Main Streamlit application
├── app_backup_20261026.py          # Backup of original app.py
├── data/
│   ├── user_transactions.csv       # Persistent transaction storage
│   ├── budgets.json                # Budget configuration
│   └── settings.json               # User preferences
├── ml/
│   ├── data/
│   │   ├── merchant_knowledge_base.csv
│   │   ├── user_feedback.csv
│   │   └── processed_transactions_v3*.csv
│   ├── models/
│   │   ├── svm_model_v3_domain.pkl
│   │   └── tfidf_vectorizer_v3_domain.pkl
│   └── src/
│       ├── hybrid_classifier.py    # Feedback → Merchant KB → ML pipeline
│       └── user_feedback.py        # Feedback persistence
├── tests/
│   └── test_spendulum.py           # 58 automated tests
└── requirements.txt
```

---

## Data Storage

- **Transactions** — `data/user_transactions.csv` (CSV, survives restarts)
- **Budgets** — `data/budgets.json` (per-category monthly limits)
- **Settings** — `data/settings.json` (currency, date format)
- **Feedback** — `ml/data/user_feedback.csv` (correction history)
- **Models** — `ml/models/` (pre-trained, not modified by the app)

All data is stored **locally on this computer**. No network access required.

---

## AI Classifier

The hybrid classifier follows this priority order:

1. **User Feedback** — exact cleaned-description matches from saved corrections
2. **Merchant Knowledge Base** — 44 known Indian merchants (`ml/data/merchant_knowledge_base.csv`)
3. **ML Model** — TF-IDF + SVM (v3 domain model, trained on synthetic Indian transaction data)

Confidence levels:
- **High** — SVM margin ≥ 0.50, or matched via feedback/merchant KB
- **Medium** — SVM margin ≥ 0.20
- **Low** — SVM margin < 0.20 → flags for user review

> **Model caveat:** The SVM model was trained on synthetic data. The confidence score is an SVM decision margin, not a calibrated probability. Accuracy on real unseen transactions may differ from synthetic-data evaluation.

---

## Running Tests

```
venv\Scripts\python.exe -m pytest tests\ -v
```

58 tests covering: classifier accuracy, transaction key deduplication, CSV normalization, user feedback persistence, analytics calculations, budget logic, CRUD operations.

---

## Supported Categories

Bills · Education · Entertainment · Food · Fuel · Groceries · Healthcare · Shopping · Transport · Travel

---

## CSV Import Format

Accepted column names (case-insensitive):

| Field | Accepted Names |
|---|---|
| Description | description, narration, merchant, transaction, details, particulars |
| Amount | amount, value, transaction amount |
| Date | date, transaction date, posted date, value date |
| Type | type, transaction type (values: Expense / Income) |

Invalid rows are reported but do not stop the import. Duplicates are detected by content fingerprint and skipped.
