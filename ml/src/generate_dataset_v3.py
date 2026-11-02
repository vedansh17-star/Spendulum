import pandas as pd
import random
from datetime import datetime, timedelta


# ==========================================
# CONFIGURATION
# ==========================================

random.seed(42)

TOTAL_PER_CATEGORY = 1000


# ==========================================
# MERCHANTS BY CATEGORY
# ==========================================

categories = {

    "Food": [
        "SWIGGY",
        "ZOMATO",
        "MCDONALDS",
        "KFC",
        "DOMINOS",
        "PIZZA HUT",
        "BURGER KING",
        "SUBWAY",
        "HALDIRAMS",
        "BIKANERVALA",
        "FRESHMENU",
        "BARBEQUE NATION",
        "FAASOS",
        "BOX8",
        "WOW MOMO"
    ],

    "Groceries": [
        "DMART",
        "SPENCERS",
        "RELIANCE SMART",
        "BIGBASKET",
        "BLINKIT",
        "ZEPTO",
        "STAR BAZAAR",
        "NATURES BASKET",
        "EASYDAY",
        "MORE SUPERMARKET",
        "JIOMART",
        "METRO CASH CARRY"
    ],

    "Transport": [
        "UBER",
        "OLA",
        "RAPIDO",
        "NAMMA YATRI",
        "MERU CABS",
        "AUTO RICKSHAW",
        "DELHI METRO",
        "MUMBAI METRO",
        "NAGPUR METRO",
        "INDRIVE"
    ],

    "Fuel": [
        "INDIAN OIL",
        "HPCL",
        "BPCL",
        "BHARAT PETROLEUM",
        "SHELL",
        "NAYARA",
        "ESSAR OIL",
        "ADANI PETROL",
        "PETROL PUMP",
        "FUEL STATION"
    ],

    "Shopping": [
        "AMAZON",
        "FLIPKART",
        "MYNTRA",
        "AJIO",
        "TATA CLIQ",
        "MEESHO",
        "DECATHLON",
        "UNIQLO",
        "NYKAA",
        "CROMA",
        "RELIANCE DIGITAL",
        "VIVO STORE",
        "NIKE"
    ],

    "Entertainment": [
        "NETFLIX",
        "AMAZON PRIME",
        "HOTSTAR",
        "SONYLIV",
        "JIOHOTSTAR",
        "YOUTUBE",
        "SPOTIFY",
        "GAANA",
        "PVR CINEMA",
        "INOX",
        "BOOKMYSHOW",
        "STEAM"
    ],

    "Bills": [
        "MSEDCL",
        "ELECTRICITY BILL",
        "AIRTEL",
        "JIO RECHARGE",
        "VODAFONE",
        "BSNL",
        "DTH RECHARGE",
        "BROADBAND BILL",
        "WIFI RECHARGE",
        "ADANI ELECTRICITY"
    ],

    "Healthcare": [
        "1MG",
        "APOLLO PHARMACY",
        "PHARMEASY",
        "NETMEDS",
        "PRACTO",
        "DR LAL PATHLABS",
        "FORTIS",
        "MAX HOSPITAL",
        "MEDPLUS"
    ],

    "Education": [
        "COURSERA",
        "UDEMY",
        "UNACADEMY",
        "BYJUS",
        "NPTEL",
        "PEARSON",
        "COACHING FEES",
        "BOOK STORE",
        "EXAM FEES",
        "COLLEGE FEES"
    ],

    "Travel": [
        "MAKEMYTRIP",
        "GOIBIBO",
        "CLEARTRIP",
        "IXIGO",
        "IRCTC",
        "VISTARA",
        "AIR INDIA",
        "INDIGO",
        "AKASA AIR",
        "OYO HOTELS",
        "HOTEL BOOKING"
    ]
}


# ==========================================
# PAYMENT PREFIXES
# ==========================================

prefixes = [
    "UPI",
    "GPAY",
    "PHONEPE",
    "PAYTM",
    "HDFC",
    "ICICI",
    "SBI",
    "AXIS",
    "KOTAK",
    "POS"
]


# ==========================================
# TRANSACTION FORMATS
# ==========================================

formats = [

    "{prefix}-{merchant}-{ref}",

    "{prefix}/{merchant}/{ref}",

    "{prefix}*{merchant}*{ref}",

    "{prefix} {merchant} PAYMENT {ref}",

    "{merchant} ORDER {ref}",

    "{merchant} TXN {ref}",

    "UPI/{merchant}/{ref}",

    "POS*{merchant}*{ref}",

    "{merchant} ONLINE PAYMENT",

    "{prefix} {merchant}"
]


# ==========================================
# AMOUNT RANGES
# ==========================================

amount_ranges = {

    "Food": (100, 2500),
    "Groceries": (200, 6000),
    "Transport": (20, 1500),
    "Fuel": (300, 5000),
    "Shopping": (300, 15000),
    "Entertainment": (50, 3000),
    "Bills": (100, 8000),
    "Healthcare": (100, 20000),
    "Education": (100, 50000),
    "Travel": (500, 30000)
}


# ==========================================
# RANDOM DATE
# ==========================================

def random_date():

    start_date = datetime(2025, 1, 1)

    end_date = datetime(2026, 8, 1)

    days = (end_date - start_date).days

    random_days = random.randint(
        0,
        days
    )

    return (
        start_date +
        timedelta(days=random_days)
    ).strftime("%Y-%m-%d")


# ==========================================
# GENERATE DATASET
# ==========================================

transactions = []

transaction_number = 1


for category, merchants in categories.items():

    min_amount, max_amount = amount_ranges[category]

    for _ in range(TOTAL_PER_CATEGORY):

        merchant = random.choice(merchants)

        prefix = random.choice(prefixes)

        ref = random.randint(
            10000,
            999999
        )

        transaction_format = random.choice(
            formats
        )

        description = transaction_format.format(
            prefix=prefix,
            merchant=merchant,
            ref=ref
        )

        amount = random.randint(
            min_amount,
            max_amount
        )

        transactions.append({

            "transaction_id":
                f"TXN{transaction_number:06d}",

            "date":
                random_date(),

            "description":
                description,

            "amount":
                amount,

            "category":
                category
        })

        transaction_number += 1


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(
    transactions
)


# ==========================================
# SHUFFLE DATASET
# ==========================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


# ==========================================
# SAVE DATASET
# ==========================================

output_path = (
    "ml/data/transactions_v3.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# OUTPUT
# ==========================================

print(
    "\n========== DATASET V3 GENERATED =========="
)

print(
    f"Total transactions: {len(df)}"
)


print(
    "\n========== CATEGORY DISTRIBUTION =========="
)

print(
    df["category"].value_counts()
)


print(
    "\n========== SAMPLE TRANSACTIONS =========="
)

print(
    df.head(10).to_string(
        index=False
    )
)