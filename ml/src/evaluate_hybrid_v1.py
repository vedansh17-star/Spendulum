import pandas as pd

from sklearn.metrics import accuracy_score

from hybrid_classifier import predict_category


# ==========================================
# 1. LOAD EVALUATION DATA
# ==========================================

df = pd.read_csv(
    "ml/data/hybrid_evaluation_v1.csv"
)


# ==========================================
# 2. RUN HYBRID PREDICTIONS
# ==========================================

results = []

for _, row in df.iterrows():

    result = predict_category(
        row["description"]
    )

    results.append({

    "description": row["description"],

    "actual_category": row["category"],

    "predicted_category": result["category"],

    "expected_source": row["expected_source"],

    "actual_source": result["source"],

    "matched_merchant": result["matched_merchant"],

    "cleaned_description": result["cleaned_text"],

    "confidence": result["confidence"],

    "confidence_score": result["confidence_score"],

    "action": result["action"]

})


results_df = pd.DataFrame(
    results
)
# ==========================================
# FLAG LOW-CONFIDENCE PREDICTIONS
# ==========================================

results_df["needs_user_confirmation"] = (

    results_df["confidence"]
    ==
    "Low"

)


# ==========================================
# 3. OVERALL ACCURACY
# ==========================================

overall_accuracy = accuracy_score(

    results_df["actual_category"],

    results_df["predicted_category"]

)


print(
    "\n========== HYBRID EVALUATION V1 =========="
)

print(
    f"Total transactions: "
    f"{len(results_df)}"
)

print(
    f"Overall Accuracy: "
    f"{overall_accuracy:.4f}"
)


# ==========================================
# 4. SOURCE ROUTING ACCURACY
# ==========================================

routing_accuracy = accuracy_score(

    results_df["expected_source"],

    results_df["actual_source"]

)


print(
    "\n========== ROUTING ACCURACY =========="
)

print(
    f"Routing Accuracy: "
    f"{routing_accuracy:.4f}"
)


# ==========================================
# 5. KNOWLEDGE BASE PERFORMANCE
# ==========================================

kb_results = results_df[
    results_df["actual_source"]
    ==
    "Merchant Knowledge Base"
]


if len(kb_results) > 0:

    kb_accuracy = accuracy_score(

        kb_results["actual_category"],

        kb_results["predicted_category"]

    )

    print(
        "\n========== KNOWLEDGE BASE PERFORMANCE =========="
    )

    print(
        f"Transactions handled: "
        f"{len(kb_results)}"
    )

    print(
        f"Accuracy: "
        f"{kb_accuracy:.4f}"
    )


# ==========================================
# 6. ML FALLBACK PERFORMANCE
# ==========================================

ml_results = results_df[
    results_df["actual_source"]
    ==
    "Machine Learning Model"
]


if len(ml_results) > 0:

    ml_accuracy = accuracy_score(

        ml_results["actual_category"],

        ml_results["predicted_category"]

    )

    print(
        "\n========== ML FALLBACK PERFORMANCE =========="
    )

    print(
        f"Transactions handled: "
        f"{len(ml_results)}"
    )

    print(
        f"Accuracy: "
        f"{ml_accuracy:.4f}"
    )
# ==========================================
# 7. CONFIDENCE ANALYSIS
# ==========================================

print(
    "\n========== CONFIDENCE ANALYSIS =========="
)

ml_confidence_results = results_df[
    results_df["actual_source"]
    ==
    "Machine Learning Model"
].copy()


ml_confidence_results["correct"] = (

    ml_confidence_results["actual_category"]
    ==
    ml_confidence_results["predicted_category"]

)


print(
    "\nConfidence distribution:"
)

print(
    ml_confidence_results[
        "confidence"
    ].value_counts()
)


print(
    "\nCorrect predictions by confidence:"
)

print(
    pd.crosstab(

        ml_confidence_results["confidence"],

        ml_confidence_results["correct"]

    )
)


print(
    "\nAverage confidence score:"
)

print(

    ml_confidence_results.groupby(
        "correct"
    )["confidence_score"].mean()

)


# ==========================================
# 7. DISPLAY ALL RESULTS
# ==========================================

print(
    "\n========== ALL RESULTS =========="
)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 8. SHOW ERRORS
# ==========================================

errors = results_df[

    results_df["actual_category"]
    !=
    results_df["predicted_category"]

]


print(
    "\n========== MISCLASSIFICATIONS =========="
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

                "actual_category",

                "predicted_category",

                "actual_source"

            ]

        ].to_string(
            index=False
        )

    )

# ==========================================
# LOW-CONFIDENCE TRANSACTIONS
# ==========================================

low_confidence = results_df[

    results_df["needs_user_confirmation"]
]


print(
    "\n========== LOW-CONFIDENCE TRANSACTIONS =========="
)


if len(low_confidence) == 0:

    print(
        "No low-confidence transactions."
    )

else:

    print(

        low_confidence[

            [

                "description",

                "actual_category",

                "predicted_category",

                "confidence",

                "confidence_score"

            ]

        ].to_string(
            index=False
        )

    )
# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df.to_csv(

    "ml/data/hybrid_evaluation_v1_results.csv",

    index=False

)


print(
    "\n========== RESULTS SAVED =========="
)

print(
    "Saved at: "
    "ml/data/hybrid_evaluation_v1_results.csv"
)