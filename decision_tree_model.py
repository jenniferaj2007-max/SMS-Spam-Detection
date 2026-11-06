import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load preprocessed data
X_train = load_npz("data/X_train_tfidf.npz")
X_test = load_npz("data/X_test_tfidf.npz")

y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Train Decision Tree model
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=30,
    min_samples_split=5
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Decision Tree Results")
print("---------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/decision_tree_model.pkl")

print("\nDecision Tree Model Saved Successfully!")