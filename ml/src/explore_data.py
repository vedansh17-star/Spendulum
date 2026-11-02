import pandas as pd

# Load dataset
df = pd.read_csv("ml/data/transactions.csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== CATEGORY COUNTS ==========")
print(df["category"].value_counts())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== BASIC STATISTICS ==========")
print(df.describe())

print("\n========== AVERAGE AMOUNT BY CATEGORY ==========")
print(df.groupby("category")["amount"].mean())
