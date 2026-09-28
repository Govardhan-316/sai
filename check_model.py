import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./hybrid_bert_model"

print("Loading dataset...")
df = pd.read_csv("WELFake_Dataset.csv")

df = df[["title", "text", "label"]].dropna()

# Take 10 Fake + 10 Real examples
fake = df[df["label"] == 0].sample(10, random_state=42)
real = df[df["label"] == 1].sample(10, random_state=42)

test_df = pd.concat([fake, real]).sample(frac=1, random_state=42)

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

correct = 0

print("\n==============================")
print("MODEL TEST - 20 ARTICLES")
print("==============================\n")

for i, row in enumerate(test_df.itertuples(), 1):

    news = str(row.title) + " " + str(row.text)

    prompt = (
        "Classify the following news article as real or fake. "
        "Use contextual language patterns and the knowledge-guided hints. "
        "News: " + news
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities))

    # WELFake: 0 = Fake, 1 = Real
    predicted_label = "Fake" if predicted_class == 0 else "Real"

    actual_label = "Fake" if int(row.label) == 0 else "Real"

    confidence = float(probabilities[predicted_class]) * 100

    if predicted_label == actual_label:
        correct += 1
        result = "CORRECT"
    else:
        result = "WRONG"

    print(
        f"{i:02d}. Actual={actual_label:<5} "
        f"Predicted={predicted_label:<5} "
        f"Confidence={confidence:6.2f}% "
        f"[{result}]"
    )

accuracy = (correct / len(test_df)) * 100

print("\n==============================")
print("FINAL RESULT")
print("==============================")
print(f"Correct predictions : {correct}/20")
print(f"Test accuracy       : {accuracy:.2f}%")
print("==============================")