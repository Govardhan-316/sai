
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATHS AND MODEL SELECTION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    ("hybrid_bert_model_v5", 192),
    ("hybrid_bert_model_v4", 128),
    ("hybrid_bert_model_v3", 128),
    ("hybrid_bert_model_v2", 128),
    ("hybrid_bert_model", 128),
]

MODEL_DIR = None
MAX_LENGTH = 128

for folder_name, default_length in MODEL_CANDIDATES:
    candidate = os.path.join(BASE_DIR, folder_name)

    if os.path.isfile(os.path.join(candidate, "config.json")):
        MODEL_DIR = candidate
        MAX_LENGTH = default_length
        break

if MODEL_DIR is None:
    raise FileNotFoundError(
        "No saved model found. Check hybrid_bert_model_v5 "
        "or the other model folders beside predict.py."
    )

# Read maximum sequence length from the training report, if available.
training_info = os.path.join(MODEL_DIR, "training_info.txt")

if os.path.isfile(training_info):
    try:
        with open(training_info, "r", encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("maximum sequence length:"):
                    MAX_LENGTH = int(line.split(":", 1)[1].strip())
                    break
    except (OSError, ValueError):
        pass

print("=" * 60)
print("Loading Fake News Detection Model")
print("Selected model folder:", MODEL_DIR)
print("Maximum sequence length:", MAX_LENGTH)
print("=" * 60)

# ============================================================
# LOAD MODEL
# ============================================================

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

except Exception as error:
    print("ERROR: Could not load the saved model.")
    print("Model folder:", MODEL_DIR)
    print("Details:", error)
    raise

device = torch.device("cpu")
model.to(device)

# Verified mapping used by the current model.
# Do NOT reverse these labels.
FAKE_CLASS_ID = 0
REAL_CLASS_ID = 1

print("Display mapping: class 0 = Fake, class 1 = Real")
print("Model loaded successfully.")
print("=" * 60)


# ============================================================
# SUSPICIOUS-LANGUAGE ANALYSIS
# These are warning signals, NOT proof of fake news.
# ============================================================

def analyze_suspicious_language(text):
    text_lower = re.sub(r"\s+", " ", str(text or "").lower()).strip()

    if not text_lower:
        return []

    categories = {
        "Sensational Language": [
            "breaking news",
            "shocking",
            "unbelievable",
            "miracle",
            "revolutionary",
            "incredible",
            "secret",
            "dramatic discovery",
        ],

        "Unverified Authority": [
            "experts say",
            "scientists claim",
            "researchers claim",
            "officials say",
            "sources say",
            "according to experts",
            "according to researchers",
            "according to scientists",
            "nasa confirms",
            "scientists confirm",
            "officials confirm",
        ],

        "Uncertainty / Allegation": [
            "reportedly",
            "allegedly",
            "supposedly",
            "claims that",
            "claim that",
            "it is believed",
            "it is said",
            "rumored",
            "rumour",
        ],

        "Conspiracy-style Wording": [
            "they don't want you to know",
            "hidden truth",
            "cover up",
            "secret government",
            "conspiracy",
            "hidden from the public",
            "destroy all evidence",
            "government is hiding",
            "government has hidden",
        ],

        "Urgency / Manipulation": [
            "act now",
            "share immediately",
            "share this news immediately",
            "must see",
            "before it's deleted",
            "before it is deleted",
            "urgent",
            "don't ignore",
        ],

        "Absolute Claim": [
            "100% true",
            "100% guaranteed",
            "guaranteed",
            "everyone knows",
            "proof that",
            "will definitely",
            "no doubt",
            "every citizen",
            "all schools worldwide",
            "everyone is guaranteed",
        ],

        "Extraordinary Scientific Claim": [
            "sun will turn green",
            "sun turns green",
            "moon is made of chocolate",
            "make people invisible",
            "become invisible",
            "see through walls",
            "read minds",
            "superhuman",
            "superpowers",
            "sun will disappear",
            "earth will stop rotating",
            "humans can breathe underwater",
        ],

        "Unusual Government or Public Announcement": [
            "all schools worldwide will remain closed",
            "all schools will remain closed",
            "every citizen will receive",
            "money will be transferred automatically",
            "stay indoors during this event",
            "without any application",
            "one month",
        ],
    }

    results = []

    for category, phrases in categories.items():
        found = [
            phrase for phrase in phrases
            if phrase in text_lower
        ]

        if found:
            results.append({
                "category": category,
                "phrases": found,
            })

    return results


# ============================================================
# INPUT FORMAT
# Must match the current training input format.
# ============================================================

def create_prediction_input(news, title=""):
    news = re.sub(r"\s+", " ", str(news or "")).strip()
    return f"News Article: {news}"


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_news(news, title=""):
    news = re.sub(r"\s+", " ", str(news or "")).strip()

    if not news:
        print("ERROR: Please enter a news article.")
        return "Unknown", 0.0

    text = create_prediction_input(news, title=title)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    class_0_probability = float(probabilities[FAKE_CLASS_ID].item())
    class_1_probability = float(probabilities[REAL_CLASS_ID].item())

    fake_probability = class_0_probability
    real_probability = class_1_probability

    predicted_class = int(torch.argmax(probabilities).item())

    if predicted_class == FAKE_CLASS_ID:
        label = "Fake"
        confidence = fake_probability * 100
    else:
        label = "Real"
        confidence = real_probability * 100

    confidence = round(confidence, 2)

    print()
    print("=" * 60)
    print("BERT NEWS ANALYSIS")
    print("=" * 60)
    print("Model folder     :", MODEL_DIR)
    print("Maximum length   :", MAX_LENGTH)
    print("Raw class 0      :", f"{class_0_probability * 100:.2f}%")
    print("Raw class 1      :", f"{class_1_probability * 100:.2f}%")
    print("Fake probability :", f"{fake_probability * 100:.2f}%")
    print("Real probability :", f"{real_probability * 100:.2f}%")
    print("Predicted class  :", predicted_class)
    print("Final prediction :", label)
    print("Confidence       :", f"{confidence:.2f}%")
    print("=" * 60)

    return label, confidence


# ============================================================
# CONSOLE TEST MODE
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("FAKE NEWS DETECTION TEST")
    print("=" * 60)
    print("Paste the news article on one line, then press ENTER.")
    print()

    news = input().strip()

    if not news:
        print("ERROR: No news article entered.")
        raise SystemExit(1)

    label, confidence = predict_news(news)
    suspicious = analyze_suspicious_language(news)

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print("BERT Prediction :", label)
    print("Model confidence:", f"{confidence:.2f}%")

    print()
    print("=" * 60)
    print("SUPPORTING LINGUISTIC ANALYSIS")
    print("=" * 60)

    if suspicious:
        for item in suspicious:
            print("\n" + item["category"])

            for phrase in item["phrases"]:
                print("  -", phrase)
    else:
        print("No listed warning patterns detected.")

    print()
    print("IMPORTANT:")
    print("- Linguistic warnings do not prove an article is fake.")
    print("- BERT confidence is not a guarantee of factual accuracy.")
    print("- Verify important claims using reliable sources.")
    print("=" * 60)

