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
# 2. CLEAN TRANSACTION FUNCTION
# ==========================================

def clean_transaction(text):

    # Convert to lowercase
    text = text.lower()

    # Remove long transaction/reference numbers
    text = re.sub(
        r"\b\d{4,}\b",
        " ",
        text
    )

    # Replace special characters with spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 3. CREATE CLEANED DESCRIPTION
# ==========================================

df["cleaned_description"] = (
    df["description"]
    .apply(clean_transaction)
)


# ==========================================
# 4. ORIGINAL VS CLEANED
# ==========================================

print(
    "\n========== ORIGINAL VS CLEANED =========="
)

comparison = df[
    [
        "description",
        "cleaned_description",
        "category"
    ]
].head(20)

print(
    comparison.to_string(
        index=False
    )
)


# ==========================================
# 5. DATA QUALITY CHECKS
# ==========================================

print(
    "\n========== MISSING VALUES =========="
)

print(
    df.isnull().sum()
)


print(
    "\n========== EMPTY CLEANED DESCRIPTIONS =========="
)

empty_descriptions = (
    df["cleaned_description"]
    .str.strip()
    .eq("")
    .sum()
)

print(
    empty_descriptions
)


print(
    "\n========== DUPLICATE ROWS =========="
)

print(
    df.duplicated().sum()
)


# ==========================================
# 6. CATEGORY DISTRIBUTION
# ==========================================

print(
    "\n========== CATEGORY DISTRIBUTION =========="
)

print(
    df["category"].value_counts()
)


# ==========================================
# 7. SAVE PROCESSED DATASET
# ==========================================

output_path = Path(
    "ml/data/processed_transactions_v3.csv"
)

df.to_csv(
    output_path,
    index=False
)


print(
    "\n========== PREPROCESSING COMPLETE =========="
)

print(
    f"Processed dataset saved at: {output_path}"
)

print(
    f"Total transactions processed: {len(df)}"
)