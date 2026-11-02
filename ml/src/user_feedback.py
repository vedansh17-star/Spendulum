import pandas as pd
from pathlib import Path


# ==========================================
# USER FEEDBACK FILE
# ==========================================

FEEDBACK_FILE = Path(__file__).resolve().parents[2] / "ml" / "data" / "user_feedback.csv"


# ==========================================
# SAVE USER CORRECTION
# ==========================================

def save_feedback(

    description,

    cleaned_description,

    predicted_category,

    corrected_category

):

    feedback = {

        "description": description,

        "cleaned_description":
        cleaned_description,

        "predicted_category":
        predicted_category,

        "corrected_category":
        corrected_category

    }


    # ==========================================
    # CREATE FILE IF IT DOES NOT EXIST
    # ==========================================

    if not FEEDBACK_FILE.exists():

        df = pd.DataFrame(
            [feedback]
        )

        df.to_csv(

            FEEDBACK_FILE,

            index=False

        )


    # ==========================================
    # APPEND TO THE CORRECTION HISTORY
    #
    # The hybrid classifier uses the latest matching
    # correction, while consolidation counts repeated
    # corrections before promoting a shared rule.
    # ==========================================

    else:

        pd.DataFrame(
            [feedback]
        ).to_csv(

            FEEDBACK_FILE,

            mode="a",

            header=False,

            index=False

        )


    print(

        "User feedback saved successfully."

    )
