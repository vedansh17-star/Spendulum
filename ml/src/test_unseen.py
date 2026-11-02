import pandas as pd
import re
import joblib
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD SAVED MODEL AND VECTORIZER
# ==========================================

model_path = Path("ml/models/svm_model.pkl")
vectorizer_path = Path("ml/models/tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ==========================================
# 2. LOAD UNSEEN DATASET
# ==========================================

df = pd.read_csv(
    "ml/data/unseen_transactions.csv"
)


# ==========================================
# 3. PREPROCESSING FUNCTION
# ==========================================

def clean_transaction(text):

    text = text.lower()

    text = re.sub(
        r"\b\d{4,}\b",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 4. CLEAN TRANSACTIONS
# ==========================================

df["cleaned_description"] = (
    df["description"]
    .apply(clean_transaction)
)


# ==========================================
# 5. VECTORIZE
# ==========================================

X_unseen = vectorizer.transform(
    df["cleaned_description"]
)


# ==========================================
# 6. PREDICT
# ==========================================

predictions = model.predict(
    X_unseen
)

df["predicted_category"] = predictions


# ==========================================
# 7. ACCURACY
# ==========================================

accuracy = accuracy_score(
    df["category"],
    predictions
)

print("\n========== HARD UNSEEN TEST ==========")

print(
    f"Total transactions: {len(df)}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ==========================================
# 8. CLASSIFICATION REPORT
# ==========================================

print(
    "\n========== CLASSIFICATION REPORT =========="
)

print(
    classification_report(
        df["category"],
        predictions,
        zero_division=0
    )
)


# ==========================================
# 9. ALL PREDICTIONS
# ==========================================

print(
    "\n========== UNSEEN PREDICTIONS =========="
)

results = df[
    [
        "cleaned_description",
        "category",
        "predicted_category"
    ]
]

print(
    results.to_string(index=False)
)


# ==========================================
# 10. ERRORS ONLY
# ==========================================

errors = df[
    df["category"]
    !=
    df["predicted_category"]
]

print(
    "\n========== MISCLASSIFIED TRANSACTIONS =========="
)

if len(errors) == 0:

    print(
        "No misclassified transactions."
    )

else:

    print(
        errors[
            [
                "description",
                "category",
                "predicted_category"
            ]
        ].to_string(index=False)
    )