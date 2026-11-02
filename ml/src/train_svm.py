import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("ml/data/processed_transactions.csv")


# =====================================================
# 2. INPUT AND TARGET
# =====================================================

X = df["cleaned_description"]
y = df["category"]


# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# 4. TF-IDF FEATURE EXTRACTION
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF FEATURES")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# =====================================================
# 5. VIEW TF-IDF VOCABULARY
# =====================================================

feature_names = vectorizer.get_feature_names_out()

print("\nNumber of features:", len(feature_names))
print("\nFirst 20 features:")
print(feature_names[:20])


# =====================================================
# 6. TRAIN SVM CLASSIFIER
# =====================================================

model = LinearSVC()

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


# =====================================================
# 7. MODEL EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))