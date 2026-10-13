
import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "WELFake_Dataset.csv"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./hybrid_bert_model_v3"

SEED = 42
MAX_ROWS = 4000
MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS = 3
LEARNING_RATE = 2e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("FAKE NEWS DETECTION - MODEL V2")
print("=" * 65)
print("Device:", device)
print("Output:", OUTPUT_DIR)

# Never overwrite an existing V2 model accidentally.
if os.path.exists(os.path.join(OUTPUT_DIR, "config.json")):
    raise FileExistsError(
        f"{OUTPUT_DIR} already contains a saved model. "
        "Choose a new OUTPUT_DIR before training again."
    )

# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

if not os.path.isfile(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(
    DATASET_PATH,
    engine="python",
    on_bad_lines="skip"
)

required_columns = {"title", "text", "label"}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required dataset columns: {sorted(missing)}"
    )

print("\nOriginal dataset shape:", df.shape)
print("Original label counts:\n", df["label"].value_counts(dropna=False))

df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()
df["label"] = df["label"].astype(int)

df["title"] = df["title"].fillna("").astype(str).str.strip()
df["text"] = df["text"].fillna("").astype(str).str.strip()

# Remove rows with neither a title nor an article.
df = df[
    (df["title"].str.len() > 0) |
    (df["text"].str.len() > 0)
].copy()

# Remove duplicate title/article pairs.
df = df.drop_duplicates(
    subset=["title", "text"]
).copy()

# Remove empty strings represented as "nan" if present.
df = df[
    (df["title"].str.lower() != "nan") |
    (df["text"].str.lower() != "nan")
].copy()

fake_df = df[df["label"] == 0]
real_df = df[df["label"] == 1]

if fake_df.empty or real_df.empty:
    raise ValueError("Dataset must contain both label 0 and label 1.")

samples_per_class = min(
    MAX_ROWS // 2,
    len(fake_df),
    len(real_df)
)

if samples_per_class < 10:
    raise ValueError("Not enough examples in each class.")

# Balanced subset for this training run.
df = pd.concat(
    [
        fake_df.sample(n=samples_per_class, random_state=SEED),
        real_df.sample(n=samples_per_class, random_state=SEED)
    ],
    ignore_index=True
)

df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("\nClean balanced dataset:", len(df))
print("Fake (0):", int((df["label"] == 0).sum()))
print("Real (1):", int((df["label"] == 1).sum()))

# ============================================================
# CONSISTENT TEXT FORMAT
# ============================================================


def create_input(row):
    article = str(row["text"]).strip()

    if article.lower() == "nan":
        article = ""

    return f"News Article: {article}".strip()

df["input_text"] = df.apply(create_input, axis=1)

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Keep the final test set separate from model selection.
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df["label"]
)

# 15% of the complete dataset for validation.
validation_fraction = 0.15 / 0.85

train_df, val_df = train_test_split(
    train_val_df,
    test_size=validation_fraction,
    random_state=SEED,
    stratify=train_val_df["label"]
)

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Final test:", len(test_df))

# ============================================================
# HUGGING FACE DATASETS
# ============================================================

def to_dataset(frame):
    selected = frame[["input_text", "label"]].copy()
    return Dataset.from_pandas(selected, preserve_index=False)

train_dataset = to_dataset(train_df)
val_dataset = to_dataset(val_df)
test_dataset = to_dataset(test_df)

# ============================================================
# TOKENIZATION
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(
    tokenize_function, batched=True
)
val_dataset = val_dataset.map(
    tokenize_function, batched=True
)
test_dataset = test_dataset.map(
    tokenize_function, batched=True
)

for dataset in (train_dataset, val_dataset, test_dataset):
    dataset = dataset.remove_columns(["input_text"])
    # Column cleanup is performed below on each dataset.

train_dataset = train_dataset.remove_columns(["input_text"])
val_dataset = val_dataset.remove_columns(["input_text"])
test_dataset = test_dataset.remove_columns(["input_text"])

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# MODEL AND LABEL MAPPING
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "Fake", 1: "Real"},
    label2id={"Fake": 0, "Real": 1}
)

# ============================================================
# VALIDATION METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
    fp16=False,
    dataloader_pin_memory=False,
    seed=SEED,
    data_seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\nStarting training. CPU training may take a long time.")
trainer.train()

# ============================================================
# FINAL TEST - USED ONLY AFTER MODEL SELECTION
# ============================================================

print("\nEvaluating on the untouched test split...")

test_output = trainer.predict(test_dataset)
test_logits = test_output.predictions
test_predictions = np.argmax(test_logits, axis=-1)
true_labels = test_df["label"].to_numpy()

print("\nFINAL TEST CLASSIFICATION REPORT")
report = classification_report(
    true_labels,
    test_predictions,
    labels=[0, 1],
    target_names=["Fake", "Real"],
    digits=4,
    zero_division=0
)
print(report)

cm = confusion_matrix(
    true_labels,
    test_predictions,
    labels=[0, 1]
)

print("\nCONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print("                 Pred Fake  Pred Real")
print(f"Actual Fake      {cm[0, 0]:9d} {cm[0, 1]:10d}")
print(f"Actual Real      {cm[1, 0]:9d} {cm[1, 1]:10d}")

test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        test_predictions,
        average="macro",
        zero_division=0
    )
)

test_accuracy = accuracy_score(true_labels, test_predictions)

# ============================================================
# SAVE V2 MODEL
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(
    os.path.join(OUTPUT_DIR, "training_info.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write("FAKE NEWS DETECTION - MODEL V2\n")
    f.write(f"Base model: {MODEL_NAME}\n")
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Balanced dataset size: {len(df)}\n")
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Validation samples: {len(val_df)}\n")
    f.write(f"Final test samples: {len(test_df)}\n")
    f.write(f"Maximum sequence length: {MAX_LENGTH}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Device: {device}\n\n")
    f.write("FINAL TEST METRICS\n")
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Macro precision: {test_precision:.4f}\n")
    f.write(f"Macro recall: {test_recall:.4f}\n")
    f.write(f"Macro F1: {test_f1:.4f}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write(report)
    f.write("\nCONFUSION MATRIX (actual rows, predicted columns)\n")
    f.write(str(cm))
    f.write("\n\nLabel mapping: 0 = Fake, 1 = Real\n")

print("\nTraining and evaluation completed.")
print("New model saved to:", OUTPUT_DIR)
print("Original model was not overwritten.")
print("Label mapping: 0 = Fake, 1 = Real")
