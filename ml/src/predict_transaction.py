from hybrid_classifier import predict_category

# ==========================================
# 4. TEST TRANSACTIONS
# ==========================================

test_transactions = [

    "UPI-SWIGGY-83921",

    "POS*DMART*72819",

    "PHONEPE-UBER-92831",

    "HDFC-NETFLIX-19283",

    "UPI-HPCL-83921",

    "PAYTM-MSEDCL-72819",

    "UPI-1MG-91827",

    "GPAY-CLEARTRIP-82931",

    "PHONEPE-COURSERA-72819",

    "UPI-MYNTRA-83921"
]


print("\n========== SPENDULUM PREDICTION TEST ==========")

for transaction in test_transactions:
    result = predict_category(transaction)

    print("\nOriginal Transaction:")
    print(transaction)

    print("Cleaned Transaction:")
    print(result["cleaned_text"])

    print("Predicted Category:")
    print(result["category"])

    print("-" * 40)