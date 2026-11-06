import joblib

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load trained models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl")
}

print("SMS Spam Detection System")
print("-------------------------")

while True:
    message = input("\nEnter an SMS message (or type 'exit' to stop): ")

    if message.lower() == "exit":
        print("Spam Detection System Closed.")
        break

    # Convert message into TF-IDF features
    message_tfidf = vectorizer.transform([message])

    # Predict using all three models
    for name, model in models.items():
        prediction = model.predict(message_tfidf)[0]

        result = "SPAM" if prediction == 1 else "HAM (Normal Message)"

        print(f"{name}: {result}")