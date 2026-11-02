import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    "ml/data/processed_transactions_v2.csv"
)

X = df["cleaned_description"]
y = df["category"]


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ==========================================
# 3. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("\n========== TF-IDF ==========")
print(f"Training shape: {X_train_tfidf.shape}")
print(f"Testing shape: {X_test_tfidf.shape}")


# ==========================================
# 4. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 5. PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 6. ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy:.4f}")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 8. SHOW SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Transaction": X_test.values,
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n========== SAMPLE PREDICTIONS ==========")

print(
    results.head(20).to_string(
        index=False
    )
)