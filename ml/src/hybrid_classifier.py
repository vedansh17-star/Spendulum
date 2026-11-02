import pandas as pd
import re
import joblib

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
DATA_DIR = PROJECT_ROOT / "ml" / "data"


# ==========================================
# 1. LOAD ML MODEL
# ==========================================

model = joblib.load(
    MODELS_DIR / "svm_model_v3_domain.pkl"
)

vectorizer = joblib.load(
    MODELS_DIR / "tfidf_vectorizer_v3_domain.pkl"
)


# ==========================================
# 2. LOAD MERCHANT KNOWLEDGE BASE
# ==========================================

merchant_df = pd.read_csv(
    DATA_DIR / "merchant_knowledge_base.csv"
)

merchant_rules = {}

for _, row in merchant_df.iterrows():

    merchant = row["merchant"].lower()

    category = row["category"]

    merchant_rules[merchant] = category

# ==========================================
# LOAD USER FEEDBACK
# ==========================================

feedback_file = Path(
    DATA_DIR / "user_feedback.csv"
)


feedback_rules = {}


if feedback_file.exists():

    feedback_df = pd.read_csv(
        feedback_file
    )


    for _, row in feedback_df.iterrows():

        cleaned_description = str(
            row["cleaned_description"]
        ).lower()


        corrected_category = (
            row["corrected_category"]
        )


        feedback_rules[
            cleaned_description
        ] = corrected_category
# ==========================================
# 3. DOMAIN NOISE WORDS
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
# 4. CLEAN TRANSACTION
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

    words = text.split()

    meaningful_words = []

    for word in words:

        if word not in noise_words:

            meaningful_words.append(
                word
            )

    return " ".join(
        meaningful_words
    )


# ==========================================
# 5. MERCHANT MATCHING
# ==========================================

def find_known_merchant(cleaned_text):

    for merchant, category in merchant_rules.items():

        if merchant in cleaned_text:

            return category, merchant

    return None, None
# ==========================================
# USER FEEDBACK MATCHING
# ==========================================

def find_feedback_match(cleaned_text):

    if cleaned_text in feedback_rules:

        return feedback_rules[
            cleaned_text
        ]

    return None

# ==========================================
# 6. HYBRID PREDICTION
# ==========================================

def predict_category(transaction):

    cleaned_text = clean_transaction(transaction)

    # ==========================================
    # 1. CHECK USER FEEDBACK
    #
    # A user correction is more specific than a shared
    # merchant default, so it must take priority.
    # ==========================================

    feedback_category = find_feedback_match(
        cleaned_text
    )

    if feedback_category is not None:

        return {

            "category": feedback_category,

            "source": "User Feedback",

            "matched_merchant": None,

            "cleaned_text": cleaned_text,

            "confidence": "High",

            "confidence_score": 1.0,

            "action": "Prediction Accepted"

        }

    # ==========================================
    # 2. CHECK MERCHANT KNOWLEDGE BASE
    # ==========================================

    category, merchant = find_known_merchant(
        cleaned_text
    )

    if category is not None:

        return {
            "category": category,
            "source": "Merchant Knowledge Base",
            "matched_merchant": merchant,
            "cleaned_text": cleaned_text,
            "confidence": "High",
            "confidence_score": 1.0,
            "action": "Prediction Accepted"
        }
    
    # ==========================================
    # 3. ML PREDICTION
    # ==========================================

    X = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(X)[0]


    # ==========================================
    # 3. GET SVM DECISION SCORES
    # ==========================================

    decision_scores = model.decision_function(X)[0]

    classes = model.classes_


    # ==========================================
    # 4. FIND TOP TWO SCORES
    # ==========================================

    sorted_indices = decision_scores.argsort()

    best_index = sorted_indices[-1]

    second_best_index = sorted_indices[-2]

    best_score = decision_scores[best_index]

    second_best_score = decision_scores[
        second_best_index
    ]


    # ==========================================
    # 5. CONFIDENCE MARGIN
    # ==========================================

    confidence_margin = (
        best_score - second_best_score
    )


    # ==========================================
    # 6. ASSIGN CONFIDENCE LEVEL
    # ==========================================

    if confidence_margin >= 0.50:

        confidence = "High"

    elif confidence_margin >= 0.20:

        confidence = "Medium"

    else:

        confidence = "Low"
    # ==========================================
# 7. DECIDE WHAT TO DO WITH THE PREDICTION
# ==========================================

    if confidence == "Low":

        action = "User Confirmation Required"

    else:

        action = "Prediction Accepted"


    # ==========================================
    # 7. RETURN RESULT
    # ==========================================

    return {
        "category": prediction,
        "source": "Machine Learning Model",
        "matched_merchant": None,
        "cleaned_text": cleaned_text,
        "confidence": confidence,
        "confidence_score": round(
            float(confidence_margin),
            4
        ),
        "action": action
    }
# ==========================================
# 7. TEST EXAMPLES
# ==========================================

if __name__ == "__main__":

    test_transactions = [

        "UPI-CHAAYOS-83921",

        "PHONEPE-HEALTHIANS-72819",

        "POS-SKECHERS-91827",

        "GPAY-YATRA-83921",

        "UPI-UNKNOWN-MERCHANT-12345",

        "UPI-CITY-CAB-918273",

        "PHONEPE-CLOTHING-OUTLET-728491",

        "GPAY-MOVIE-TICKET-839201"
    ]


    print(
        "\n========== HYBRID CLASSIFIER TEST =========="
    )


    for transaction in test_transactions:

        result = predict_category(
            transaction
        )

        print(
            "\nTransaction:",
            transaction
        )

        print(
            "Cleaned:",
            result["cleaned_text"]
        )

        print(
            "Predicted Category:",
            result["category"]
        )

        print(
            "Prediction Source:",
            result["source"]
        )


        print(
            "Confidence:",
            result["confidence"]
        )

        print(
            "Confidence Score:",
            result["confidence_score"]
        )
        print(
            "Action:",
             result["action"]
        )

        print(
            "Matched Merchant:",
            result["matched_merchant"]
        )
