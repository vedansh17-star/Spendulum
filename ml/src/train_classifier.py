import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("ml/data/processed_transactions.csv")


# ==========================================
# 2. DEFINE INPUT AND TARGET
# ==========================================

X = df["cleaned_description"]
y = df["category"]


# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE TF-IDF VECTORIZER
# ==========================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==========================================
# 5. CREATE ML MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model.fit(X_train_tfidf, y_train)


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)
print("\n========== PREDICTIONS ==========")

results = pd.DataFrame({
    "Transaction": X_test,
    "Actual": y_test,
    "Predicted": y_pred
})

print(results.to_string(index=False))


# ==========================================
# 8. EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))