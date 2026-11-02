import pandas as pd
import re
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# ==========================================
# 1. LOAD V3-B MODEL AND VECTORIZER
# ==========================================

model_path = Path(
    "ml/models/svm_model_v3_domain.pkl"
)

vectorizer_path = Path(
    "ml/models/tfidf_vectorizer_v3_domain.pkl"
)

model = joblib.load(model_path)

vectorizer = joblib.load(
    vectorizer_path
)


# ==========================================
# 2. LOAD STRICT UNSEEN DATASET
# ==========================================

df = pd.read_csv(
    "ml/data/strict_unseen_v3.csv"
)


# ==========================================
# 3. DOMAIN NOISE WORDS
# Must match training preprocessing
# ==========================================

noise_words = {
    "upi",
    "gpay",
    "phonepe",
    "paytm",

    "hdfc",
    "icici",
    "sbi",
    "axis",
    "kotak",

    "pos",

    "txn",
    "transaction",

    "payment",
    "online",
    "order"
}


# ==========================================
# 4. DOMAIN-AWARE CLEANING
# ==========================================

def clean_transaction(text):

    # Convert to lowercase
    text = text.lower()

    # Remove long transaction/reference numbers
    text = re.sub(
        r"\b\d{4,}\b",
        " ",
        text
    )

    # Replace special characters
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Split into words
    words = text.split()

    # Remove domain noise
    meaningful_words = []

    for word in words:

        if word not in noise_words:
            meaningful_words.append(word)

    # Join again
    text = " ".join(
        meaningful_words
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# 5. PREPROCESS STRICT UNSEEN DATA
# ==========================================

df["domain_cleaned_description"] = (
    df["description"]
    .apply(clean_transaction)
)


# ==========================================
# 6. TRANSFORM
# ==========================================

X_unseen = vectorizer.transform(
    df["domain_cleaned_description"]
)


# ==========================================
# 7. PREDICT
# ==========================================

predictions = model.predict(
    X_unseen
)

df["predicted_category"] = predictions


# ==========================================
# 8. ACCURACY
# ==========================================

accuracy = accuracy_score(
    df["category"],
    predictions
)


print(
    "\n========== STRICT UNSEEN V3-B TEST =========="
)

print(
    f"Total transactions: {len(df)}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ==========================================
# 9. CLASSIFICATION REPORT
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
# 10. ALL PREDICTIONS
# ==========================================

results = df[
    [
        "domain_cleaned_description",
        "category",
        "predicted_category"
    ]
]

print(
    "\n========== ALL STRICT UNSEEN PREDICTIONS =========="
)

print(
    results.to_string(
        index=False
    )
)


# ==========================================
# 11. MISCLASSIFICATIONS
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
        ].to_string(
            index=False
        )
    )


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results_path = Path(
    "ml/data/strict_unseen_v3_domain_results.csv"
)

df.to_csv(
    results_path,
    index=False
)

print(
    "\n========== RESULTS SAVED =========="
)

print(
    f"Results saved at: {results_path}"
)