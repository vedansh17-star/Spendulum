import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Load processed dataset
df = pd.read_csv("ml/data/processed_transactions.csv")


# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()


# Convert transaction descriptions into numerical vectors
X = vectorizer.fit_transform(df["cleaned_description"])


print("========== TF-IDF SHAPE ==========")
print(X.shape)


print("\n========== VOCABULARY ==========")
print(vectorizer.get_feature_names_out())


print("\n========== FIRST TRANSACTION ==========")
print(df["cleaned_description"].iloc[0])


print("\n========== FIRST TF-IDF VECTOR ==========")
print(X[0].toarray())