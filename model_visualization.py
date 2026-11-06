import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
df = pd.read_csv("results/model_comparison.csv")

# Select evaluation metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Create comparison graph
df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    width=0.8
)

plt.title("SMS Spam Detection - Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save graph
plt.savefig("results/model_performance_comparison.png", dpi=300)
plt.show()

print("Model Comparison Graph Saved Successfully!")