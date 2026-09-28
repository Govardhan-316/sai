import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./hybrid_bert_model"

print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.eval()

print("BERT model loaded successfully.")


# ---------------------------------------------------------
# SUSPICIOUS LANGUAGE / KNOWLEDGE ANALYSIS
# ---------------------------------------------------------

def analyze_suspicious_language(text):

    text_lower = text.lower()

    categories = {
        "Absolute Claim": [
            "100% true",
            "100% guaranteed",
            "guaranteed",
            "always",
            "never",
            "everyone knows",
            "proof that",
            "all diseases"
        ],

        "Sensational Language": [
            "shocking",
            "breaking news",
            "breaking",
            "unbelievable",
            "miracle",
            "secret",
            "you won't believe"
        ],

        "Conspiracy-style Wording": [
            "they don't want you to know",
            "hidden truth",
            "cover up",
            "secret government",
            "conspiracy",
            "hidden from the public",
            "destroy all evidence"
        ],

        "Urgency / Manipulation": [
            "act now",
            "share immediately",
            "share this news immediately",
            "must see",
            "before it's deleted",
            "before it is deleted",
            "urgent"
        ],

        "Unverified Authority": [
            "experts say",
            "scientists say",
            "officials say",
            "sources say",
            "researchers claim",
            "researchers say",
            "authorities are allegedly"
        ]
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
                "phrases": found
            })

    return results


# ---------------------------------------------------------
# HYBRID PREDICTION
# ---------------------------------------------------------

def predict_news(news):

    news = re.sub(r"\s+", " ", news).strip()

    # BERT prompt
    prompt = (
        "Classify the following news article as real or fake. "
        "Use contextual language patterns and knowledge-guided hints. "
        "News: " + news
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # BERT prediction
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]

    bert_fake_probability = float(probabilities[0])
    bert_real_probability = float(probabilities[1])

    bert_class = int(torch.argmax(probabilities))

    # -----------------------------------------------------
    # KNOWLEDGE / SUSPICIOUS LANGUAGE ANALYSIS
    # -----------------------------------------------------

    suspicious = analyze_suspicious_language(news)

    category_count = len(suspicious)

    phrase_count = sum(
        len(item.get("phrases", []))
        for item in suspicious
    )

    # -----------------------------------------------------
    # HYBRID DECISION
    # -----------------------------------------------------
    #
    # Strong suspicious language can override a BERT
    # "Real" prediction when multiple categories are present.
    #
    # This is a project-level knowledge-guided rule and
    # should not be described as factual verification.
    # -----------------------------------------------------

    if category_count >= 3 or phrase_count >= 5:

        predicted_class = 0

        # Confidence based on strength of suspicious signals
        confidence = min(
            99.0,
            70.0 + (category_count * 5.0) + (phrase_count * 2.0)
        )

    else:

        predicted_class = bert_class

        confidence = float(
            probabilities[predicted_class]
        ) * 100

    # WELFake:
    # 0 = Fake
    # 1 = Real

    label = "Fake" if predicted_class == 0 else "Real"

    confidence = round(confidence, 2)

    return label, confidence


# ---------------------------------------------------------
# TEST MODE
# ---------------------------------------------------------

if __name__ == "__main__":

    news = input("\nPaste the news article:\n")

    label, confidence = predict_news(news)

    print("\n==============================")
    print("Prediction :", label)
    print("Confidence :", f"{confidence:.2f}%")
    print("==============================")

    suspicious = analyze_suspicious_language(news)

    print("\nSuspicious Language:")

    if suspicious:

        for item in suspicious:
            print(
                f"- {item['category']}: "
                f"{', '.join(item['phrases'])}"
            )

    else:
        print("No strong suspicious patterns detected.")