import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

SAMPLES_PER_CATEGORY = 200


# ==================================================
# CATEGORY DATA
# ==================================================

CATEGORY_DATA = {

    "Food": {
        "merchants": [
            "SWIGGY",
            "ZOMATO",
            "DOMINOS",
            "KFC",
            "MCDONALDS",
            "STARBUCKS",
            "CAFE COFFEE DAY",
            "PIZZA HUT",
            "BURGER KING",
            "SUBWAY"
        ],

        "amount_range": (100, 2500)
    },


    "Groceries": {
        "merchants": [
            "DMART",
            "BIGBASKET",
            "BLINKIT",
            "ZEPTO",
            "RELIANCE SMART",
            "MORE SUPERMARKET",
            "SPENCERS",
            "GROCERY STORE",
            "VEGETABLE MARKET",
            "MILK DAIRY"
        ],

        "amount_range": (100, 5000)
    },


    "Transport": {
        "merchants": [
            "UBER",
            "OLA",
            "RAPIDO",
            "NAGPUR METRO",
            "DELHI METRO",
            "AUTO RICKSHAW",
            "CITY BUS",
            "RED BUS",
            "IRCTC"
        ],

        "amount_range": (20, 3000)
    },


    "Fuel": {
        "merchants": [
            "INDIAN OIL",
            "HPCL",
            "BPCL",
            "BHARAT PETROLEUM",
            "RELIANCE PETROL",
            "SHELL"
        ],

        "amount_range": (200, 5000)
    },


    "Shopping": {
        "merchants": [
            "AMAZON",
            "FLIPKART",
            "MYNTRA",
            "AJIO",
            "TATA CLiQ",
            "RELIANCE TRENDS",
            "CROMA",
            "NYKAA",
            "H&M"
        ],

        "amount_range": (200, 15000)
    },


    "Entertainment": {
        "merchants": [
            "NETFLIX",
            "PRIME VIDEO",
            "SPOTIFY",
            "PVR CINEMAS",
            "INOX",
            "BOOKMYSHOW",
            "YOUTUBE PREMIUM",
            "HOTSTAR"
        ],

        "amount_range": (100, 3000)
    },


    "Bills": {
        "merchants": [
            "JIO",
            "AIRTEL",
            "VI",
            "BSNL",
            "MSEDCL",
            "MAHAVITARAN",
            "TATA POWER",
            "BROADBAND",
            "DTH RECHARGE"
        ],

        "amount_range": (100, 10000)
    },


    "Healthcare": {
        "merchants": [
            "APOLLO PHARMACY",
            "MEDPLUS",
            "NETMEDS",
            "1MG",
            "FORTIS HOSPITAL",
            "MAX HOSPITAL",
            "PHARMACY",
            "DIAGNOSTIC LAB"
        ],

        "amount_range": (100, 20000)
    },


    "Education": {
        "merchants": [
            "UDEMY",
            "COURSERA",
            "UNACADEMY",
            "BYJUS",
            "COLLEGE FEES",
            "EXAM FEES",
            "BOOK STORE",
            "ONLINE COURSE"
        ],

        "amount_range": (200, 50000)
    },


    "Travel": {
        "merchants": [
            "MAKEMYTRIP",
            "GOIBIBO",
            "YATRA",
            "IRCTC TRAVEL",
            "AIR INDIA",
            "INDIGO",
            "VISTARA",
            "OYO ROOMS"
        ],

        "amount_range": (500, 50000)
    }
}


# ==================================================
# TRANSACTION DESCRIPTION GENERATOR
# ==================================================

def generate_description(merchant):

    transaction_id = random.randint(10000, 999999)

    templates = [

        f"UPI-{merchant}-{transaction_id}",

        f"UPI/{merchant}/{transaction_id}",

        f"POS*{merchant}*{transaction_id}",

        f"PHONEPE-{merchant}-{transaction_id}",

        f"GPAY-{merchant}-{transaction_id}",

        f"PAYTM-{merchant}-{transaction_id}",

        f"{merchant} PAYMENT",

        f"{merchant} ONLINE PAYMENT",

        f"{merchant} INDIA",

        f"{merchant} ORDER {transaction_id}",

        f"{merchant} TRANSACTION",

        f"HDFC-{merchant}-{transaction_id}",

        f"SBI-{merchant}-{transaction_id}",

        f"ICICI-{merchant}-{transaction_id}",

        f"AXIS-{merchant}-{transaction_id}"

    ]

    return random.choice(templates)


# ==================================================
# RANDOM DATE GENERATOR
# ==================================================

def generate_date():

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 8, 1)

    days_difference = (end_date - start_date).days

    random_days = random.randint(0, days_difference)

    return (
        start_date + timedelta(days=random_days)
    ).strftime("%Y-%m-%d")


# ==================================================
# GENERATE DATASET
# ==================================================

transactions = []

transaction_counter = 1


for category, details in CATEGORY_DATA.items():

    merchants = details["merchants"]
    min_amount, max_amount = details["amount_range"]

    for _ in range(SAMPLES_PER_CATEGORY):

        merchant = random.choice(merchants)

        description = generate_description(merchant)

        amount = random.randint(
            min_amount,
            max_amount
        )

        date = generate_date()

        transaction = {
            "transaction_id": f"TXN{transaction_counter:06d}",
            "date": date,
            "description": description,
            "amount": amount,
            "category": category
        }

        transactions.append(transaction)

        transaction_counter += 1


# ==================================================
# CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(transactions)


# ==================================================
# SHUFFLE DATA
# ==================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==================================================
# SAVE DATASET
# ==================================================

output_path = Path("ml/data/transactions_v2.csv")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_path,
    index=False
)


# ==================================================
# OUTPUT
# ==================================================

print("\n========== DATASET GENERATED ==========")

print(f"Total transactions: {len(df)}")

print("\n========== CATEGORY DISTRIBUTION ==========")

print(df["category"].value_counts())

print("\n========== FIRST 10 ROWS ==========")

print(df.head(10).to_string(index=False))

print(
    f"\nDataset saved successfully at:\n{output_path}"
)