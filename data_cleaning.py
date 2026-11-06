import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Original Dataset Shape:", df.shape)

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate messages
df.drop_duplicates(inplace=True)

# Convert messages to lowercase
df["message"] = df["message"].str.lower().str.strip()

# Standardize labels
df["label"] = df["label"].str.lower().str.strip()

# Keep only valid labels
df = df[df["label"].isin(["ham", "spam"])]

# Reset index
df.reset_index(drop=True, inplace=True)

print("\nCleaned Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Save cleaned dataset
df.to_csv("data/cleaned_sms_data.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")