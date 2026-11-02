import pandas as pd
import re
from pathlib import Path


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("ml/data/transactions_v2.csv")


# ==========================================
# 2. CLEAN TEXT FUNCTION
# ==========================================

def clean_transaction(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove long numeric transaction/reference IDs
    # Example: 82931, 728194
    text = re.sub(r"\b\d{4,}\b", " ", text)

    # Replace special characters with spaces
    # Example: UPI-SWIGGY → UPI SWIGGY
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# 3. APPLY PREPROCESSING
# ==========================================

df["cleaned_description"] = df["description"].apply(
    clean_transaction
)


# ==========================================
# 4. DISPLAY BEFORE AND AFTER
# ==========================================

print("\n========== ORIGINAL VS CLEANED ==========")

comparison = df[
    ["description", "cleaned_description", "category"]
].head(20)

print(comparison.to_string(index=False))


# ==========================================
# 5. CHECK FOR EMPTY CLEANED DESCRIPTIONS
# ==========================================

empty_rows = (
    df["cleaned_description"]
    .str.strip()
    .eq("")
    .sum()
)

print("\n========== EMPTY CLEANED DESCRIPTIONS ==========")
print(empty_rows)


# ==========================================
# 6. SAVE PROCESSED DATASET
# ==========================================

output_path = Path(
    "ml/data/processed_transactions_v2.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 7. FINAL OUTPUT
# ==========================================

print("\n========== PREPROCESSING COMPLETE ==========")

print(f"Total transactions: {len(df)}")

print(f"\nProcessed dataset saved at:\n{output_path}")