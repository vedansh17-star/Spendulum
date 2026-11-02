import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DOMAIN-PROCESSED DATASET
# ==========================================

df = pd.read_csv(
    "ml/data/processed_transactions_v3_domain.csv"
)

X = df["domain_cleaned_description"]
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
# 3. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print("\n========== TF-IDF ==========")
print(f"Training shape: {X_train_tfidf.shape}")
print(f"Testing shape: {X_test_tfidf.shape}")

print(
    f"Number of features: "
    f"{len(vectorizer.get_feature_names_out())}"
)


# ==========================================
# 4. TRAIN SVM
# ==========================================

model = LinearSVC(
    random_state=42
)

print("\n========== TRAINING SVM V3-B ==========")

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 5. PREDICT
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 6. PERFORMANCE
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== SVM V3-B PERFORMANCE ==========")
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
# 8. MISCLASSIFICATIONS
# ==========================================

results = pd.DataFrame({
    "Transaction": X_test.values,
    "Actual": y_test.values,
    "Predicted": predictions
})

errors = results[
    results["Actual"] != results["Predicted"]
]

print(
    "\n========== MISCLASSIFIED TRANSACTIONS =========="
)

if len(errors) == 0:
    print("No misclassified transactions.")
else:
    print(errors.to_string(index=False))


# ==========================================
# 9. SAVE MODEL
# ==========================================

models_dir = Path("ml/models")

models_dir.mkdir(
    parents=True,
    exist_ok=True
)


model_path = models_dir / "svm_model_v3_domain.pkl"

vectorizer_path = (
    models_dir /
    "tfidf_vectorizer_v3_domain.pkl"
)


joblib.dump(
    model,
    model_path
)

joblib.dump(
    vectorizer,
    vectorizer_path
)


print("\n========== V3-B MODEL SAVED ==========")

print(
    f"Model saved at: {model_path}"
)

print(
    f"Vectorizer saved at: "
    f"{vectorizer_path}"
)