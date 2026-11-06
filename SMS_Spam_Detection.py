import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst Five Messages:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

print("\nSpam and Ham Message Counts:")
print(df["label"].value_counts())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nLabel Categories:")
print(df["label"].unique())

print("\nMessage Length Statistics:")
print(df["message"].str.len().describe())

print("\nData Understanding Completed!")