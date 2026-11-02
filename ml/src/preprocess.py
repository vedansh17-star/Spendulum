import pandas as pd
import re


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Replace special characters with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load the dataset
df = pd.read_csv("ml/data/transactions.csv")


# Apply cleaning to every transaction description
df["cleaned_description"] = df["description"].apply(clean_text)


# Display original and cleaned descriptions
print("========== ORIGINAL VS CLEANED ==========")
print(df[["description", "cleaned_description"]].head(10))


# Save the processed dataset
df.to_csv("ml/data/processed_transactions.csv", index=False)

print("\nProcessed dataset saved successfully!")