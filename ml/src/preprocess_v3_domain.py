import pandas as pd
import re
from pathlib import Path


# ==========================================
# 1. LOAD DATASET V3
# ==========================================

df = pd.read_csv(
    "ml/data/transactions_v3.csv"
)


# ==========================================
# 2. DOMAIN-SPECIFIC NOISE WORDS
# ==========================================

noise_words = {
    "upi",
    "gpay",
    "phonepe",
    "paytm",

    "hdfc",
    "icici",
    "sbi",
    "axis",
    "kotak",

    "pos",

    "txn",
    "transaction",

    "payment",
    "online",
    "order"
}


# ==========================================
# 3. CLEAN TRANSACTION
# ==========================================

def clean_transaction(text):

    # Convert to lowercase
    text = text.lower()

    # Remove long reference numbers
    text = re.sub(
        r"\b\d{4,}\b",
        " ",
        text
    )

    # Replace special characters
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Split into individual words
    words = text.split()

    # Remove transaction noise
    meaningful_words = []

    for word in words:

        if word not in noise_words:

            meaningful_words.append(
                word
            )

    # Join words again
    text = " ".join(
        meaningful_words
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 4. APPLY CLEANING
# ==========================================

df["domain_cleaned_description"] = (
    df["description"]
    .apply(clean_transaction)
)


# ==========================================
# 5. COMPARE CLEANING
# ==========================================

print(
    "\n========== ORIGINAL VS DOMAIN CLEANED =========="
)

comparison = df[
    [
        "description",
        "domain_cleaned_description",
        "category"
    ]
].head(30)

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================
# 6. QUALITY CHECKS
# ==========================================

print(
    "\n========== EMPTY DESCRIPTIONS =========="
)

print(
    df[
        "domain_cleaned_description"
    ].eq("")
    .sum()
)


print(
    "\n========== CATEGORY DISTRIBUTION =========="
)

print(
    df["category"]
    .value_counts()
)


# ==========================================
# 7. SAVE DATASET
# ==========================================

output_path = Path(
    "ml/data/processed_transactions_v3_domain.csv"
)

df.to_csv(
    output_path,
    index=False
)


print(
    "\n========== DOMAIN PREPROCESSING COMPLETE =========="
)

print(
    f"Saved at: {output_path}"
)

print(
    f"Total transactions: {len(df)}"
)