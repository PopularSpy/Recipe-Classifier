# train_and_save.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# File paths
RAW_PATH = "Tagged_Food_Recipes.csv"
META_PATH = "recipes_meta.csv"

# Read CSV with latin1 encoding
df = pd.read_csv(RAW_PATH, encoding="latin1")
print("✅ Dataset loaded successfully with latin1 encoding")

# Make sure required columns exist
required_cols = ["Title", "Ingredients", "Instructions", "Tags", "Image_Name"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Missing column in CSV: {col}")

# Create combined column
df["combined"] = (
    df["Title"].fillna("") + " " +
    df["Ingredients"].fillna("") + " " +
    df["Instructions"].fillna("") + " " +
    df["Tags"].fillna("")
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["combined"])

# Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(X)

# Save vectorizer & model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("nn_model.pkl", "wb") as f:
    pickle.dump(nn_model, f)

# Save processed dataset
df.to_csv(META_PATH, index=False)

print("✅ Training complete. Files saved:")
print("- vectorizer.pkl")
print("- nn_model.pkl")
print(f"- {META_PATH}")
