import pandas as pd
import joblib


from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
# 4. TRAIN LINEAR SVM
# ==========================================

model = LinearSVC(
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 6. MODEL PERFORMANCE
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== SVM PERFORMANCE ==========")
print(f"Accuracy: {accuracy:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 7. SAMPLE PREDICTIONS
# ==========================================

results = pd.DataFrame({
    "Transaction": X_test.values,
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n========== SAMPLE PREDICTIONS ==========")

print(results.head(20).to_string(index=False))


# ==========================================
# 8. MISCLASSIFIED TRANSACTIONS
# ==========================================

errors = results[
    results["Actual"] != results["Predicted"]
]

print("\n========== MISCLASSIFIED TRANSACTIONS ==========")

if len(errors) == 0:
    print("No misclassified transactions.")

else:
    print(errors.to_string(index=False))
# ==========================================
# 9. SAVE MODEL AND VECTORIZER
# ==========================================

model_path = Path("ml/models/svm_model.pkl")

vectorizer_path = Path(
    "ml/models/tfidf_vectorizer.pkl"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


print("\n========== MODEL SAVED ==========")

print(f"SVM model saved at: {model_path}")

print(
    f"TF-IDF vectorizer saved at: {vectorizer_path}"
)