"""
Spendulum Database Layer (SQLite).
Robust local database for transactions, budgets, settings, and demo seeding.
"""

import sqlite3
import pandas as pd
import hashlib
from pathlib import Path
from uuid import uuid4
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "spendulum.db"
CSV_FALLBACK = PROJECT_ROOT / "data" / "user_transactions.csv"

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    
    # 1. Transactions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        transaction_key TEXT UNIQUE,
        date TEXT,
        description TEXT,
        amount REAL,
        type TEXT,
        category TEXT,
        confidence TEXT,
        confidence_score REAL,
        prediction_source TEXT,
        matched_merchant TEXT,
        review_status TEXT,
        is_demo INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Budgets table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS budgets (
        category TEXT PRIMARY KEY,
        monthly_limit REAL
    );
    """)

    # 3. Settings table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)

    conn.commit()

    # Seed initial budgets if empty
    cur.execute("SELECT COUNT(*) FROM budgets")
    if cur.fetchone()[0] == 0:
        default_budgets = [
            ("Housing and utilities", 18000.0),
            ("Food and groceries", 12000.0),
            ("Transport and fuel", 7000.0),
            ("Healthcare", 3000.0),
            ("Shopping", 5000.0),
            ("Entertainment", 3000.0),
            ("Education", 4000.0),
            ("Travel", 4000.0),
            ("Other expenses", 5000.0),
            ("_monthly_total", 61000.0)
        ]
        cur.executemany("INSERT INTO budgets (category, monthly_limit) VALUES (?, ?)", default_budgets)
        conn.commit()

    # Migrate from existing CSV if DB transactions table is empty
    cur.execute("SELECT COUNT(*) FROM transactions")
    if cur.fetchone()[0] == 0 and CSV_FALLBACK.exists():
        try:
            df = pd.read_csv(CSV_FALLBACK)
            if not df.empty:
                for _, row in df.iterrows():
                    tid = str(row.get("Transaction ID", "")) or uuid4().hex
                    tkey = str(row.get("Transaction Key", "")) or build_key(row)
                    d_val = str(row.get("Date", ""))
                    desc = str(row.get("Description", ""))
                    amt = float(pd.to_numeric(row.get("Amount", 0), errors="coerce") or 0.0)
                    ttype = str(row.get("Type", "Expense"))
                    cat = str(row.get("Category", "Other expenses"))
                    conf = str(row.get("Confidence", "High"))
                    cscore = float(pd.to_numeric(row.get("Confidence Score", 1.0), errors="coerce") or 1.0)
                    src = str(row.get("Prediction Source", "Machine Learning Model"))
                    merch = str(row.get("Matched Merchant", "—"))
                    rev = str(row.get("Review Status", "Prediction Accepted"))
                    
                    cur.execute("""
                    INSERT OR IGNORE INTO transactions 
                    (transaction_id, transaction_key, date, description, amount, type, category, confidence, confidence_score, prediction_source, matched_merchant, review_status, is_demo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """, (tid, tkey, d_val, desc, amt, ttype, cat, conf, cscore, src, merch, rev))
                conn.commit()
        except Exception as e:
            print("CSV migration note:", e)

    conn.close()

def build_key(row):
    d_val = str(row.get("Date") or row.get("date") or "")
    desc = str(row.get("Description") or row.get("description") or "").strip().lower()
    amt = f"{float(pd.to_numeric(row.get('Amount') or row.get('amount') or 0, errors='coerce') or 0):.2f}"
    ttype = str(row.get("Type") or row.get("type") or "Expense").strip().lower()
    fp = f"{d_val}|{desc}|{amt}|{ttype}"
    return hashlib.sha256(fp.encode("utf-8")).hexdigest()

def get_all_transactions():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY date DESC", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        # Aliases for UI backwards compatibility
        df["Transaction ID"] = df["transaction_id"]
        df["Transaction Key"] = df["transaction_key"]
        df["Date"] = df["date"]
        df["Description"] = df["description"]
        df["Amount"] = df["amount"]
        df["Type"] = df["type"]
        df["Category"] = df["category"]
        df["Confidence"] = df["confidence"]
        df["Confidence Score"] = df["confidence_score"]
        df["Prediction Source"] = df["prediction_source"]
        df["Matched Merchant"] = df["matched_merchant"]
        df["Review Status"] = df["review_status"]
    return df

def insert_transaction(record):
    conn = get_connection()
    cur = conn.cursor()
    tid = record.get("transaction_id") or record.get("Transaction ID") or uuid4().hex
    tkey = record.get("transaction_key") or record.get("Transaction Key") or build_key(record)
    d_val = str(record.get("date") or record.get("Date") or date.today())
    desc = str(record.get("description") or record.get("Description") or "").strip()
    amt = float(record.get("amount") or record.get("Amount") or 0.0)
    ttype = str(record.get("type") or record.get("Type") or "Expense")
    cat = str(record.get("category") or record.get("Category") or "Other expenses")
    conf = str(record.get("confidence") or record.get("Confidence") or "High")
    cscore = float(record.get("confidence_score") or record.get("Confidence Score") or 1.0)
    src = str(record.get("prediction_source") or record.get("Prediction Source") or "Manual Entry")
    merch = str(record.get("matched_merchant") or record.get("Matched Merchant") or "—")
    rev = str(record.get("review_status") or record.get("Review Status") or ("User Confirmation Required" if conf == "Low" else "Prediction Accepted"))
    is_demo = int(record.get("is_demo", 0))

    try:
        cur.execute("""
        INSERT INTO transactions (transaction_id, transaction_key, date, description, amount, type, category, confidence, confidence_score, prediction_source, matched_merchant, review_status, is_demo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tid, tkey, d_val, desc, amt, ttype, cat, conf, cscore, src, merch, rev, is_demo))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def update_transaction(transaction_id, updates):
    conn = get_connection()
    cur = conn.cursor()
    fields = []
    values = []
    field_map = {
        "Description": "description", "description": "description",
        "Amount": "amount", "amount": "amount",
        "Category": "category", "category": "category",
        "Type": "type", "type": "type",
        "Date": "date", "date": "date",
        "Confidence": "confidence", "confidence": "confidence",
        "Confidence Score": "confidence_score", "confidence_score": "confidence_score",
        "Prediction Source": "prediction_source", "prediction_source": "prediction_source",
        "Review Status": "review_status", "review_status": "review_status"
    }
    for k, v in updates.items():
        if k in field_map:
            fields.append(f"{field_map[k]} = ?")
            values.append(v)
    if fields:
        values.append(transaction_id)
        cur.execute(f"UPDATE transactions SET {', '.join(fields)} WHERE transaction_id = ?", values)
        conn.commit()
    conn.close()

def delete_transaction(transaction_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE transaction_id = ?", (transaction_id,))
    conn.commit()
    conn.close()

def get_budgets():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT category, monthly_limit FROM budgets")
    rows = cur.fetchall()
    conn.close()
    return {r["category"]: float(r["monthly_limit"]) for r in rows}

def save_budgets(budget_dict):
    conn = get_connection()
    cur = conn.cursor()
    for cat, lim in budget_dict.items():
        cur.execute("""
        INSERT INTO budgets (category, monthly_limit) VALUES (?, ?)
        ON CONFLICT(category) DO UPDATE SET monthly_limit = excluded.monthly_limit
        """, (cat, float(lim)))
    conn.commit()
    conn.close()

def seed_demo_profile():
    """
    Seeds a realistic Indian profile:
    Annual Salary: ₹11,50,000 (~₹95,833/month)
    Generates realistic month-to-month transactions across August, September, October 2026.
    Includes salary credits, housing rents, bills, groceries, dining, fuel, travel, shopping,
    and review queue items with realistic varied amounts so trends show dynamic financial movement.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Check if demo data already loaded
    cur.execute("SELECT COUNT(*) FROM transactions WHERE is_demo = 1")
    if cur.fetchone()[0] > 0:
        conn.close()
        return 0 # Already seeded

    demo_txns = []
    
    # Detailed variations per month to create natural, realistic financial trajectories
    month_variations = {
        "2026-08": {"electric": 2480.0, "dmart": 5230.0, "swiggy1": 740.0, "swiggy2": 580.0, "uber": 980.0, "fuel": 2400.0, "shop": 3120.0, "travel": 4200.0, "misc": 1450.0},
        "2026-09": {"electric": 3120.0, "dmart": 6150.0, "swiggy1": 920.0, "swiggy2": 810.0, "uber": 1450.0, "fuel": 2850.0, "shop": 4500.0, "travel": 2900.0, "misc": 1980.0},
        "2026-10": {"electric": 2750.0, "dmart": 5840.0, "swiggy1": 840.0, "swiggy2": 650.0, "uber": 1250.0, "fuel": 2600.0, "shop": 2499.0, "travel": 3800.0, "misc": 1650.0},
    }

    for m, v in month_variations.items():
        # Monthly Salary Credit
        demo_txns.append({
            "Date": f"{m}-01", "Description": "SALARY CREDIT - TECH CORP INDIA", "Amount": 95833.00,
            "Type": "Income", "Category": "Housing and utilities", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Rule Base", "Matched Merchant": "tech corp", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Rent / Housing
        demo_txns.append({
            "Date": f"{m}-02", "Description": "UPI-RENT PAYMENT TO LANDLORD", "Amount": 14000.00,
            "Type": "Expense", "Category": "Housing and utilities", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "rent", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Utilities / Electricity (MSEDCL)
        demo_txns.append({
            "Date": f"{m}-05", "Description": "PAYTM-MSEDCL-ELECTRICITY BILL", "Amount": v["electric"],
            "Type": "Expense", "Category": "Housing and utilities", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "msedcl", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Groceries (DMart)
        demo_txns.append({
            "Date": f"{m}-07", "Description": "POS*DMART*HYPERMARKET", "Amount": v["dmart"],
            "Type": "Expense", "Category": "Food and groceries", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "dmart", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Food & Dining (Swiggy)
        demo_txns.append({
            "Date": f"{m}-10", "Description": "UPI-SWIGGY-BANGALORE", "Amount": v["swiggy1"],
            "Type": "Expense", "Category": "Food and groceries", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "swiggy", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        demo_txns.append({
            "Date": f"{m}-18", "Description": "UPI-SWIGGY-ORDER-DINNER", "Amount": v["swiggy2"],
            "Type": "Expense", "Category": "Food and groceries", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "swiggy", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        demo_txns.append({
            "Date": f"{m}-22", "Description": "CHAAYOS TEA CAFE", "Amount": 320.00,
            "Type": "Expense", "Category": "Food and groceries", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "chaayos", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Transport & Fuel
        demo_txns.append({
            "Date": f"{m}-08", "Description": "PHONEPE-UBER-RIDES", "Amount": v["uber"],
            "Type": "Expense", "Category": "Transport and fuel", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "uber", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        demo_txns.append({
            "Date": f"{m}-15", "Description": "UPI-HPCL-PETROL PUMP", "Amount": v["fuel"],
            "Type": "Expense", "Category": "Transport and fuel", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "hpcl", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Healthcare
        demo_txns.append({
            "Date": f"{m}-12", "Description": "UPI-1MG-PHARMACY HEALTHCARE", "Amount": 1450.00,
            "Type": "Expense", "Category": "Healthcare", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "1mg", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Entertainment (Netflix & Cinema)
        demo_txns.append({
            "Date": f"{m}-04", "Description": "HDFC-NETFLIX-SUBSCRIPTION", "Amount": 649.00,
            "Type": "Expense", "Category": "Entertainment", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "netflix", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        demo_txns.append({
            "Date": f"{m}-20", "Description": "BOOKMYSHOW-CARNIVAL CINEMAS", "Amount": 880.00,
            "Type": "Expense", "Category": "Entertainment", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "carnival cinema", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Shopping (Myntra)
        demo_txns.append({
            "Date": f"{m}-16", "Description": "UPI-MYNTRA-FASHION APPAREL", "Amount": v["shop"],
            "Type": "Expense", "Category": "Shopping", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "myntra", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Education (Coursera)
        demo_txns.append({
            "Date": f"{m}-11", "Description": "PHONEPE-COURSERA-LEARNING", "Amount": 3200.00,
            "Type": "Expense", "Category": "Education", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "coursera", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Travel (Cleartrip)
        demo_txns.append({
            "Date": f"{m}-24", "Description": "GPAY-CLEARTRIP-WEEKEND TRAVEL", "Amount": v["travel"],
            "Type": "Expense", "Category": "Travel", "Confidence": "High", "Confidence Score": 1.0,
            "Prediction Source": "Merchant Knowledge Base", "Matched Merchant": "cleartrip", "Review Status": "Prediction Accepted", "is_demo": 1
        })
        # Unresolved item needing review (Ambiguous or low confidence)
        demo_txns.append({
            "Date": f"{m}-26", "Description": f"POS PAYMENT MISC STORE {m[-2:]}", "Amount": v["misc"],
            "Type": "Expense", "Category": "Other expenses", "Confidence": "Low", "Confidence Score": 0.12,
            "Prediction Source": "Machine Learning Model", "Matched Merchant": "—", "Review Status": "User Confirmation Required", "is_demo": 1
        })

    count = 0
    for t in demo_txns:
        if insert_transaction(t):
            count += 1
            
    conn.close()
    return count

    count = 0
    for t in demo_txns:
        if insert_transaction(t):
            count += 1
            
    conn.close()
    return count

def clear_demo_data():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE is_demo = 1")
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    return deleted
