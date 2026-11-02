import pandas as pd


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("ml/data/transactions_v2.csv")


# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("\n========== FIRST 10 ROWS ==========")
print(df.head(10).to_string(index=False))


print("\n========== DATASET INFORMATION ==========")
print(df.info())


# ==========================================
# 3. CATEGORY DISTRIBUTION
# ==========================================

print("\n========== CATEGORY DISTRIBUTION ==========")
print(df["category"].value_counts())


# ==========================================
# 4. MISSING VALUES
# ==========================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ==========================================
# 5. DUPLICATE ROWS
# ==========================================

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ==========================================
# 6. BASIC AMOUNT STATISTICS
# ==========================================

print("\n========== AMOUNT STATISTICS ==========")
print(df["amount"].describe())


# ==========================================
# 7. AVERAGE AMOUNT BY CATEGORY
# ==========================================

print("\n========== AVERAGE AMOUNT BY CATEGORY ==========")

average_amount = (
    df.groupby("category")["amount"]
    .mean()
    .sort_values()
)

print(average_amount)


# ==========================================
# 8. SAMPLE TRANSACTIONS BY CATEGORY
# ==========================================

print("\n========== SAMPLE TRANSACTIONS ==========")

for category in df["category"].unique():

    sample = df[
        df["category"] == category
    ][["description", "amount"]].head(3)

    print(f"\n--- {category} ---")
    print(sample.to_string(index=False))