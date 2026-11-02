import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. LOAD PROCESSED DATASET
# ==========================================

df = pd.read_csv(
    "ml/data/processed_transactions_v2.csv"
)


# ==========================================
# 2. CREATE TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)


# ==========================================
# 3. CONVERT TEXT TO FEATURES
# ==========================================

X_tfidf = vectorizer.fit_transform(
    df["cleaned_description"]
)


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n========== DATASET SIZE ==========")
print(f"Transactions: {len(df)}")


print("\n========== TF-IDF SHAPE ==========")
print(X_tfidf.shape)


print("\n========== NUMBER OF FEATURES ==========")
print(len(vectorizer.get_feature_names_out()))


print("\n========== SAMPLE FEATURES ==========")

features = vectorizer.get_feature_names_out()

print(features[:100])


# ==========================================
# 5. SHOW EXAMPLE VECTOR
# ==========================================

print("\n========== EXAMPLE TRANSACTION ==========")

example_transaction = df.iloc[0]["cleaned_description"]

print(example_transaction)


print("\n========== EXAMPLE TF-IDF VECTOR ==========")

print(
    X_tfidf[0].toarray()
)