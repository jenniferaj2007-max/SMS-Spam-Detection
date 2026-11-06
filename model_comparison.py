import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
X_test = load_npz("data/X_test_tfidf.npz")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Load trained models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl")
}

results = []

# Evaluate all models
for name, model in models.items():
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Create comparison table
comparison_df = pd.DataFrame(results)

print("ML Model Comparison")
print("-------------------")
print(comparison_df.to_string(index=False))

# Save comparison results
comparison_df.to_csv("results/model_comparison.csv", index=False)

print("\nModel Comparison Saved Successfully!")