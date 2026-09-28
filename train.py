import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from datasets import Dataset


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "WELFake_Dataset.csv"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./hybrid_bert_model"

# Small dataset for CPU training/demo
MAX_ROWS = 2000

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading WELFake dataset...")

df = pd.read_csv(DATASET_PATH)

print("Original columns:", df.columns.tolist())
print("Original rows:", len(df))


# ============================================================
# 3. SELECT REQUIRED COLUMNS
# ============================================================

if not all(column in df.columns for column in ["title", "text", "label"]):
    raise ValueError(
        "Dataset must contain title, text and label columns."
    )

df = df[["title", "text", "label"]].copy()

print("Required columns:", df.columns.tolist())


# ============================================================
# 4. CLEAN DATA
# ============================================================

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=["label"]
)

df["label"] = df["label"].astype(int)

# Keep only Fake = 0 and Real = 1
df = df[df["label"].isin([0, 1])]

# Remove empty news articles
df = df[
    df["text"].str.strip() != ""
]

df = df.reset_index(drop=True)


# ============================================================
# 5. BALANCED SAMPLE FOR CPU
# ============================================================

print("\nCreating balanced sample...")

fake_news = df[df["label"] == 0]
real_news = df[df["label"] == 1]

number_each = min(
    MAX_ROWS // 2,
    len(fake_news),
    len(real_news)
)

fake_news = fake_news.sample(
    n=number_each,
    random_state=SEED
)

real_news = real_news.sample(
    n=number_each,
    random_state=SEED
)

df = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

# Shuffle
df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("Rows after sampling:", len(df))
print("Fake news:", len(df[df["label"] == 0]))
print("Real news:", len(df[df["label"] == 1]))


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["title"] = df["title"].apply(
    clean_text
)

df["text"] = df["text"].apply(
    clean_text
)


# ============================================================
# 7. LOAD KNOWLEDGE BASE
# ============================================================

print("\nLoading knowledge base...")

with open(
    "knowledge_base.json",
    "r",
    encoding="utf-8"
) as f:

    KB = json.load(f)


# ============================================================
# 8. KNOWLEDGE HINTS
# ============================================================

def knowledge_hints(article):

    lower = article.lower()

    claim_hits = [
        word
        for word in KB["claim_words"]
        if word.lower() in lower
    ]

    credibility_hits = [
        word
        for word in KB["credibility_words"]
        if word.lower() in lower
    ]

    hints = []

    if claim_hits:
        hints.append(
            "sensational_claim"
        )

    if credibility_hits:
        hints.append(
            "credibility_reference"
        )

    if not hints:
        hints.append(
            "neutral"
        )

    return ", ".join(hints)


# ============================================================
# 9. CREATE PROMPT
# ============================================================

def make_prompt(row):

    article = (
        row["title"] +
        " " +
        row["text"]
    ).strip()

    knowledge = knowledge_hints(
        article
    )

    prompt = (
        "Classify the following news article "
        "as real or fake. "
        "Use contextual language patterns "
        "and knowledge-guided hints. "
        f"Knowledge hints: {knowledge}. "
        f"News: {article}"
    )

    return prompt


print("\nCreating knowledge-guided prompts...")

df["prompt_text"] = df.apply(
    make_prompt,
    axis=1
)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df[["prompt_text", "label"]],
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"]
)

print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))


# ============================================================
# 11. CONVERT TO HUGGING FACE DATASET
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


# ============================================================
# 12. LOAD BERT TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 13. TOKENIZATION
# ============================================================

def tokenize(batch):

    return tokenizer(
        batch["prompt_text"],
        truncation=True,
        max_length=128
    )


print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

print("Tokenizing testing data...")

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)


# Remove original prompt
train_dataset = train_dataset.remove_columns(
    ["prompt_text"]
)

test_dataset = test_dataset.remove_columns(
    ["prompt_text"]
)


# ============================================================
# 14. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 15. LOAD BERT MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ============================================================
# 16. EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 17. PROGRESSIVE TRAINING - STAGE 1
# ============================================================

print("\n")
print("==========================================")
print("STAGE 1: CLASSIFICATION HEAD TRAINING")
print("==========================================")


# Freeze BERT layers
for param in model.bert.parameters():

    param.requires_grad = False


stage1_args = TrainingArguments(
    output_dir="./stage1",

    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=2e-4,

    eval_strategy="epoch",

    save_strategy="no",

    logging_steps=50,

    report_to="none"
)


trainer = Trainer(
    model=model,

    args=stage1_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


trainer.train()


# ============================================================
# 18. PROGRESSIVE TRAINING - STAGE 2
# ============================================================

print("\n")
print("==========================================")
print("STAGE 2: FULL BERT FINE-TUNING")
print("==========================================")


# Unfreeze BERT
for param in model.bert.parameters():

    param.requires_grad = True


stage2_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    eval_strategy="epoch",

    save_strategy="no",

    logging_steps=50,

    report_to="none"
)


trainer = Trainer(
    model=model,

    args=stage2_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


trainer.train()


# ============================================================
# 19. FINAL EVALUATION
# ============================================================

print("\n")
print("==========================================")
print("FINAL RESULTS")
print("==========================================")


metrics = trainer.evaluate()


print("\nAccuracy  :", metrics.get("eval_accuracy"))
print("Precision :", metrics.get("eval_precision"))
print("Recall    :", metrics.get("eval_recall"))
print("F1 Score  :", metrics.get("eval_f1"))


# ============================================================
# 20. SAVE MODEL
# ============================================================

print("\nSaving model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# Save label mapping
with open(
    os.path.join(
        OUTPUT_DIR,
        "label_mapping.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "0": "Fake",
            "1": "Real"
        },
        f,
        indent=2
    )


print("\n")
print("==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(
    "Model location:",
    OUTPUT_DIR
)

print("\nTraining completed!")