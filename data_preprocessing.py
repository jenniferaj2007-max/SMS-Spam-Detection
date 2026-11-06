import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sms_data.csv")

# Convert labels into numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Separate features and target
X = df["message"]
y = df["label"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save processed data
save_npz("data/X_train_tfidf.npz", X_train_tfidf)
save_npz("data/X_test_tfidf.npz", X_test_tfidf)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Training Data Shape:", X_train_tfidf.shape)
print("Testing Data Shape:", X_test_tfidf.shape)

print("\nTraining and Testing Split Completed!")
print("TF-IDF Preprocessing Completed Successfully!")