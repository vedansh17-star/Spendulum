import pandas as pd
from pathlib import Path


# ==========================================
# FILE PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEEDBACK_FILE = PROJECT_ROOT / "ml" / "data" / "user_feedback.csv"

KNOWLEDGE_BASE_FILE = PROJECT_ROOT / "ml" / "data" / "merchant_knowledge_base.csv"


# ==========================================
# CHECK FEEDBACK FILE
# ==========================================

if not FEEDBACK_FILE.exists():

    print(
        "No user feedback file found."
    )

    exit()


feedback_df = pd.read_csv(
    FEEDBACK_FILE
)


print(
    "\n========== FEEDBACK CONSOLIDATION =========="
)

print(
    f"Total feedback records: "
    f"{len(feedback_df)}"
)


# ==========================================
# GROUP IDENTICAL CORRECTIONS
# ==========================================

feedback_summary = (

    feedback_df

    .groupby(

        [
            "cleaned_description",
            "corrected_category"
        ]

    )

    .size()

    .reset_index(
        name="correction_count"
    )

)


print(
    "\n========== FEEDBACK SUMMARY =========="
)

print(
    feedback_summary.to_string(
        index=False
    )
)


# ==========================================
# CONSOLIDATION THRESHOLD
# ==========================================

MIN_CORRECTIONS = 2


# Only the most recent correction for a description can be
# promoted. This prevents contradictory historical corrections
# from adding competing merchant rules.
latest_feedback = (
    feedback_df
    .drop_duplicates(
        subset=["cleaned_description"],
        keep="last"
    )
    [["cleaned_description", "corrected_category"]]
)

approved_feedback = (
    latest_feedback
    .merge(
        feedback_summary,
        on=["cleaned_description", "corrected_category"],
        how="left"
    )
    .query("correction_count >= @MIN_CORRECTIONS")
)


print(
    "\n========== APPROVED FEEDBACK =========="
)

if len(approved_feedback) == 0:

    print(
        "No feedback has reached "
        f"the threshold of "
        f"{MIN_CORRECTIONS} corrections."
    )

else:

    print(
        approved_feedback.to_string(
            index=False
        )
    )
# ==========================================
# PROMOTE APPROVED FEEDBACK
# ==========================================

if len(approved_feedback) > 0:

    # Load existing knowledge base

    knowledge_base_df = pd.read_csv(
        KNOWLEDGE_BASE_FILE
    )


    # Track new rules

    new_rules = []


    for _, row in approved_feedback.iterrows():

        merchant = row[
            "cleaned_description"
        ]

        category = row[
            "corrected_category"
        ]


        # Check whether merchant already exists

        merchant_exists = (

            knowledge_base_df[
                "merchant"
            ]
            .str.lower()
            .eq(
                merchant.lower()
            )
            .any()

        )


        # Add only if not already present

        if not merchant_exists:

            new_rules.append({

                "merchant": merchant,

                "category": category

            })


    # Add approved feedback to knowledge base

    if len(new_rules) > 0:

        new_rules_df = pd.DataFrame(
            new_rules
        )


        knowledge_base_df = pd.concat(

            [

                knowledge_base_df,

                new_rules_df

            ],

            ignore_index=True

        )


        knowledge_base_df.to_csv(

            KNOWLEDGE_BASE_FILE,

            index=False

        )


        print(
            "\n========== PROMOTION COMPLETE =========="
        )

        print(
            f"New rules added: "
            f"{len(new_rules)}"
        )

        print(
            "\nAdded rules:"
        )

        print(
            new_rules_df.to_string(
                index=False
            )
        )


    else:

        print(
            "\n========== PROMOTION COMPLETE =========="
        )

        print(
            "No new rules were added."
        )
