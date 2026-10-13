
import pandas as pd
from sklearn.model_selection import train_test_split
from predict import predict_news

DATASET = "WELFake_Dataset.csv"
MAX_ROWS_PER_CLASS = 10000

print("Loading and preparing dataset...")

df = pd.read_csv(DATASET)

# Keep the required columns
df = df[["title", "text", "label"]].copy()

# Match the training label filtering
df["label"] = df["label"].astype(str).str.strip()
df = df[df["label"].isin(["0", "1"])].copy()
df["label"] = df["label"].astype(int)

# Match the training text cleaning
for column in ["title", "text"]:
    df[column] = df[column].fillna("").astype(str).str.strip()

df = df[~((df["title"] == "") & (df["text"] == ""))]
df = df.drop_duplicates(subset=["title", "text"])

# Match the balanced dataset preparation
fake = df[df["label"] == 0]
real = df[df["label"] == 1]

fake = fake.sample(
    n=min(MAX_ROWS_PER_CLASS, len(fake)),
    random_state=42
)
real = real.sample(
    n=min(MAX_ROWS_PER_CLASS, len(real)),
    random_state=42
)

df = pd.concat([fake, real]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Recreate the held-out test split
_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Test 10 Fake and 10 Real held-out articles
fake_test = test_df[test_df["label"] == 0].sample(
    n=min(10, len(test_df[test_df["label"] == 0])),
    random_state=42
)
real_test = test_df[test_df["label"] == 1].sample(
    n=min(10, len(test_df[test_df["label"] == 1])),
    random_state=42
)

test_df = pd.concat([fake_test, real_test]).sample(
    frac=1,
    random_state=42
)

correct = 0
total = len(test_df)

print("\n========================================")
print("SAVED MODEL TEST - HELD-OUT ARTICLES")
print("========================================")

for i, (_, row) in enumerate(test_df.iterrows(), 1):

    # Match train.py: use article text; use title only if text is empty
    news = row["text"] if row["text"] else row["title"]

    # Use the same prediction function as the application
    prediction, confidence = predict_news(news)

    actual = "Fake" if row["label"] == 0 else "Real"
    result = "CORRECT" if prediction == actual else "WRONG"

    if result == "CORRECT":
        correct += 1

    print(
        f"\n{i:02d}. Actual={actual:<5} "
        f"Predicted={prediction:<5} "
        f"Confidence={confidence:6.2f}% "
        f"[{result}]"
    )
    print("Article:", news[:180].replace("\n", " "))

accuracy = correct / total * 100 if total else 0

print("\n========================================")
print("FINAL RESULT")
print("========================================")
print(f"Correct predictions: {correct}/{total}")
print(f"Accuracy on this sample: {accuracy:.2f}%")
print("========================================")
