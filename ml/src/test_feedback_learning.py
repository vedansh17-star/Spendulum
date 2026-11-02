from hybrid_classifier import predict_category


test_transactions = [

    "UPI-CITY-CAB-918273",

    "PHONEPE-CLOTHING-OUTLET-728491",

    "GPAY-MOVIE-TICKET-839201"

]


print(
    "\n========== FEEDBACK LEARNING TEST =========="
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