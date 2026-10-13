
from flask import Flask, request, jsonify, render_template_string
import sqlite3
from datetime import datetime
import os
import re

from predict import predict_news, analyze_suspicious_language


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

DATABASE = "prediction_history.db"

# Trained model directory.
# The accuracy is read automatically from training_info.txt.
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hybrid_bert_model_v5"
)

TRAINING_INFO_FILE = os.path.join(
    MODEL_DIR,
    "training_info.txt"
)


def get_model_accuracy():
    """Read the latest saved test accuracy from the trained model files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Try all normal locations so the app works whether it is
    # started from C:\fake-news-detection or another working directory.
    candidates = [
        os.path.join(base_dir, "hybrid_bert_model_v5", "training_info.txt"),
        os.path.join(os.getcwd(), "hybrid_bert_model_v5", "training_info.txt"),
        os.path.join(base_dir, "hybrid_bert_model_v4", "training_info.txt"),
        os.path.join(os.getcwd(), "hybrid_bert_model_v4", "training_info.txt"),
        os.path.join(base_dir, "hybrid_bert_model", "training_info.txt"),
        os.path.join(os.getcwd(), "hybrid_bert_model", "training_info.txt"),
        os.path.join(base_dir, "training_info.txt"),
        os.path.join(os.getcwd(), "training_info.txt"),
    ]

    for file_path in candidates:
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Supports the exact format written by your train.py:
            # Accuracy      : 0.9899494004297498
            patterns = [
                r"(?im)^\s*Accuracy\s*:\s*([0-9]*\.?[0-9]+)",
                r"(?im)^\s*eval_accuracy\s*:\s*([0-9]*\.?[0-9]+)",
                r"(?im)^\s*accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    accuracy = float(match.group(1))

                    if 0 <= accuracy <= 1:
                        accuracy *= 100

                    accuracy = round(accuracy, 2)

                    print(
                        f"MODEL ACCURACY LOADED: {accuracy:.2f}%"
                    )
                    print(f"Accuracy source: {file_path}")
                    return accuracy

        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    # Final fallback: read Hugging Face trainer_state.json if present.
    import json

    trainer_files = [
        os.path.join(base_dir, "hybrid_bert_model_v5", "trainer_state.json"),
        os.path.join(base_dir, "hybrid_bert_model_v5", "checkpoint-525", "trainer_state.json"),
        os.path.join(os.getcwd(), "hybrid_bert_model_v5", "trainer_state.json"),
        os.path.join(os.getcwd(), "hybrid_bert_model_v5", "checkpoint-525", "trainer_state.json"),
        os.path.join(base_dir, "hybrid_bert_model_v4", "trainer_state.json"),
        os.path.join(base_dir, "hybrid_bert_model", "trainer_state.json"),
    ]

    for file_path in trainer_files:
        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            history = state.get("log_history", [])

            for entry in reversed(history):
                value = entry.get("eval_accuracy")
                if value is not None:
                    accuracy = float(value)
                    if 0 <= accuracy <= 1:
                        accuracy *= 100
                    accuracy = round(accuracy, 2)
                    print(
                        f"MODEL ACCURACY LOADED: {accuracy:.2f}%"
                    )
                    print(f"Accuracy source: {file_path}")
                    return accuracy

        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    print("WARNING: Model accuracy file was not found.")
    print("Expected: hybrid_bert_model\\training_info.txt")
    return None


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def init_database():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(
    news_text,
    prediction,
    confidence,
    explanation
):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (
            news_text,
            prediction,
            confidence,
            explanation,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        news_text,
        prediction,
        confidence,
        explanation,
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            news_text,
            prediction,
            confidence,
            explanation,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()

    conn.close()

    history = []

    for row in rows:

        history.append({
            "id": row[0],
            "news_text": row[1],
            "prediction": row[2],
            "confidence": row[3],
            "explanation": row[4],
            "created_at": row[5]
        })

    return history


def get_statistics():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM predictions"
    )

    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE LOWER(prediction) = 'real'
    """)

    real_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM predictions
        WHERE LOWER(prediction) = 'fake'
    """)

    fake_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT AVG(confidence)
        FROM predictions
    """)

    avg_confidence = cursor.fetchone()[0]

    conn.close()

    if avg_confidence is None:
        avg_confidence = 0

    return {
        "total": total,
        "real": real_count,
        "fake": fake_count,
        "average_confidence": round(
            avg_confidence,
            1
        )
    }


def clear_history():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    conn.commit()
    conn.close()


# ============================================================
# SEPARATE WARNING LAYER
# ============================================================

def get_warning_layer(text):
    """Return linguistic warning signals separately from BERT's label.

    This layer does not change the BERT prediction and does not verify truth.
    """
    try:
        suspicious = analyze_suspicious_language(text)
    except Exception as error:
        print("Warning-layer analysis error:", error)
        suspicious = []

    items = []
    categories = set()

    for item in suspicious:
        if isinstance(item, dict):
            category = str(item.get("category", "Suspicious Language"))
            phrases = item.get("phrases", [])
            if isinstance(phrases, list):
                for phrase in phrases:
                    phrase = str(phrase)
                    items.append({"category": category, "phrase": phrase})
                    categories.add(category)
        else:
            phrase = str(item)
            items.append({"category": "Suspicious Language", "phrase": phrase})
            categories.add("Suspicious Language")

    category_count = len(categories)
    if category_count >= 3:
        level = "review"
        status = "Several warning patterns found"
    elif category_count >= 1:
        level = "review"
        status = "Some warning patterns found"
    else:
        level = "none"
        status = "No listed warning patterns found"

    return {
        "count": len(items),
        "category_count": category_count,
        "level": level,
        "status": status,
        "items": items,
        "message": (
            "These are language-based warning signals only. They do not prove "
            "that an article is fake, and they do not change the BERT prediction. "
            "Verify important claims with reliable, independent sources."
        ),
    }


def get_explanation(text):
    """Backward-compatible alias used by existing app code and UI."""
    return get_warning_layer(text)


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        NewsGuard AI | Fake News Detection
    </title>


    <!-- Bootstrap 5 -->

    <link
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
        rel="stylesheet"
    >


    <!-- Bootstrap Icons -->

    <link
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"
        rel="stylesheet"
    >


    <style>

        /* ==================================================
           GLOBAL
        ================================================== */

        :root {

            --primary: #4f46e5;
            --primary-dark: #3730a3;

            --dark: #111827;
            --muted: #6b7280;

            --bg: #f5f7fb;

            --card: #ffffff;

            --border: #e5e7eb;

            --success: #16a34a;

            --danger: #dc2626;

            --warning: #d97706;
        }


        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            background:
                linear-gradient(
                    135deg,
                    #f8fafc,
                    #eef2ff
                );

            color: var(--dark);

            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
        }


        /* ==================================================
           NAVBAR
        ================================================== */

        .navbar-custom {

            background: rgba(
                255,
                255,
                255,
                0.94
            );

            backdrop-filter: blur(12px);

            border-bottom:
                1px solid var(--border);

            position: sticky;

            top: 0;

            z-index: 1000;
        }


        .brand-icon {

            width: 42px;

            height: 42px;

            border-radius: 12px;

            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #7c3aed
                );

            color: white;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 21px;

            box-shadow:
                0 8px 20px
                rgba(79,70,229,.25);
        }


        .brand-title {

            font-weight: 800;

            font-size: 19px;

            letter-spacing: -.4px;
        }


        .brand-subtitle {

            color: var(--muted);

            font-size: 11px;

            margin-top: -3px;
        }


        .nav-link {

            color: #4b5563 !important;

            font-weight: 600;

            margin-left: 10px;

            cursor: pointer;
        }


        .nav-link:hover {

            color:
                var(--primary) !important;
        }


        .online-dot {

            width: 9px;

            height: 9px;

            background:
                #22c55e;

            border-radius: 50%;

            display: inline-block;

            box-shadow:
                0 0 0 4px
                rgba(34,197,94,.12);
        }


        /* ==================================================
           MAIN
        ================================================== */

        .main-container {

            max-width: 1250px;

            margin: auto;

            padding:
                45px 20px 70px;
        }


        .hero-title {

            font-size:
                clamp(
                    30px,
                    4vw,
                    46px
                );

            font-weight: 800;

            letter-spacing: -1.5px;

            margin-bottom: 10px;
        }


        .hero-description {

            color: var(--muted);

            font-size: 16px;

            max-width: 720px;
        }


        .model-pill {

            background:
                #eef2ff;

            color:
                #4338ca;

            border:
                1px solid #c7d2fe;

            padding:
                10px 16px;

            border-radius: 30px;

            font-size: 13px;

            font-weight: 700;

            display: inline-flex;

            align-items: center;

            gap: 7px;
        }


        /* ==================================================
           STAT CARDS
        ================================================== */

        .stat-card {

            background: var(--card);

            border:
                1px solid var(--border);

            border-radius: 18px;

            padding: 22px;

            height: 100%;

            box-shadow:
                0 8px 25px
                rgba(15,23,42,.05);

            transition:
                transform .2s,
                box-shadow .2s;
        }


        .stat-card:hover {

            transform:
                translateY(-3px);

            box-shadow:
                0 15px 35px
                rgba(15,23,42,.09);
        }


        .stat-icon {

            width: 43px;

            height: 43px;

            border-radius: 12px;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 19px;

            margin-bottom: 15px;
        }


        .icon-blue {

            background: #eef2ff;

            color: #4f46e5;
        }


        .icon-green {

            background: #ecfdf5;

            color: #16a34a;
        }


        .icon-red {

            background: #fef2f2;

            color: #dc2626;
        }


        .icon-orange {

            background: #fff7ed;

            color: #ea580c;
        }


        .stat-label {

            color: var(--muted);

            font-size: 12px;

            font-weight: 700;

            text-transform:
                uppercase;

            letter-spacing: .6px;
        }


        .stat-value {

            font-size: 30px;

            font-weight: 800;

            margin-top: 4px;
        }


        /* ==================================================
           CARDS
        ================================================== */

        .dashboard-card {

            background: white;

            border:
                1px solid var(--border);

            border-radius: 20px;

            box-shadow:
                0 8px 30px
                rgba(15,23,42,.055);

            overflow: hidden;

            height: 100%;
        }


        .card-header-custom {

            padding:
                22px 25px;

            border-bottom:
                1px solid var(--border);

            background:
                rgba(249,250,251,.75);
        }


        .card-title-custom {

            font-size: 16px;

            font-weight: 800;
        }


        .card-subtitle-custom {

            font-size: 12px;

            color: var(--muted);

            margin-top: 3px;
        }


        .card-body-custom {

            padding: 25px;
        }


        /* ==================================================
           NEWS TEXTAREA
        ================================================== */

        .news-textarea {

            width: 100%;

            min-height: 260px;

            border:
                1px solid #dbe1ea;

            border-radius: 15px;

            padding: 18px;

            resize: vertical;

            outline: none;

            font-size: 14px;

            line-height: 1.7;

            transition:
                border .2s,
                box-shadow .2s;
        }


        .news-textarea:focus {

            border-color:
                var(--primary);

            box-shadow:
                0 0 0 4px
                rgba(79,70,229,.10);
        }


        .btn-primary-custom {

            background:
                linear-gradient(
                    135deg,
                    #4f46e5,
                    #6366f1
                );

            border: none;

            color: white;

            font-weight: 700;

            padding:
                11px 21px;

            border-radius: 11px;

            transition:
                transform .2s,
                box-shadow .2s;
        }


        .btn-primary-custom:hover {

            color: white;

            transform:
                translateY(-1px);

            box-shadow:
                0 8px 20px
                rgba(79,70,229,.25);
        }


        .btn-secondary-custom {

            background: white;

            border:
                1px solid var(--border);

            color: #374151;

            font-weight: 600;

            padding:
                10px 17px;

            border-radius: 11px;
        }


        .btn-secondary-custom:hover {

            background:
                #f9fafb;
        }


        /* ==================================================
           MODEL STATUS
        ================================================== */

        .status-box {

            display: flex;

            align-items: center;

            gap: 12px;

            padding: 15px;

            background:
                #f0fdf4;

            border:
                1px solid #bbf7d0;

            border-radius: 13px;

            margin-bottom: 22px;
        }


        .status-indicator {

            width: 11px;

            height: 11px;

            background:
                #22c55e;

            border-radius: 50%;

            box-shadow:
                0 0 0 5px
                rgba(34,197,94,.12);
        }


        .status-title {

            font-size: 13px;

            font-weight: 800;

            color: #166534;
        }


        .status-description {

            font-size: 11px;

            color: #4b7c5b;
        }


        .info-row {

            display: flex;

            justify-content: space-between;

            padding:
                12px 0;

            border-bottom:
                1px solid #f0f1f3;

            font-size: 13px;
        }


        .info-row:last-child {

            border-bottom: none;
        }


        .info-label {

            color: var(--muted);
        }


        .info-value {

            font-weight: 700;
        }


        /* ==================================================
           RESULT
        ================================================== */

        .result-placeholder {

            min-height: 300px;

            display: flex;

            flex-direction: column;

            align-items: center;

            justify-content: center;

            text-align: center;

            color: var(--muted);
        }


        .result-icon {

            width: 65px;

            height: 65px;

            border-radius: 20px;

            background:
                #eef2ff;

            color:
                #4f46e5;

            display: flex;

            align-items: center;

            justify-content: center;

            font-size: 28px;

            margin-bottom: 15px;
        }


        .prediction-badge {

            display: inline-flex;

            align-items: center;

            gap: 8px;

            padding:
                9px 15px;

            border-radius: 30px;

            font-size: 13px;

            font-weight: 800;
        }


        .prediction-fake {

            background:
                #fef2f2;

            color:
                #b91c1c;

            border:
                1px solid #fecaca;
        }


        .prediction-real {

            background:
                #ecfdf5;

            color:
                #15803d;

            border:
                1px solid #bbf7d0;
        }


        .confidence-number {

            font-size: 43px;

            font-weight: 900;

            letter-spacing: -1px;
        }


        .confidence-label {

            color: var(--muted);

            font-size: 12px;

            font-weight: 600;
        }


        .progress {

            height: 10px;

            border-radius: 20px;

            background:
                #eef0f4;
        }


        .progress-bar {

            border-radius: 20px;
        }


        .signal-badge {

            display: inline-block;

            padding:
                7px 11px;

            border-radius: 8px;

            background:
                #fff7ed;

            color:
                #c2410c;

            border:
                1px solid #fed7aa;

            font-size: 11px;

            font-weight: 700;

            margin:
                4px 4px 4px 0;
        }


        /* ==================================================
           PIPELINE
        ================================================== */

        .pipeline {

            display: flex;

            align-items: center;

            justify-content: space-between;

            gap: 8px;

            overflow-x: auto;

            padding:
                10px 0;
        }


        .pipeline-step {

            min-width: 120px;

            text-align: center;
        }


        .pipeline-number {

            width: 40px;

            height: 40px;

            border-radius: 50%;

            background:
                #eef2ff;

            color:
                #4f46e5;

            margin:
                0 auto 9px;

            display: flex;

            align-items: center;

            justify-content: center;

            font-weight: 800;

            font-size: 13px;
        }


        .pipeline-title {

            font-size: 12px;

            font-weight: 800;
        }


        .pipeline-arrow {

            color:
                #9ca3af;

            font-size: 20px;
        }


        /* ==================================================
           HISTORY TABLE
        ================================================== */

        .table-wrapper {

            overflow-x: auto;
        }


        .history-table {

            margin: 0;

            min-width: 700px;
        }


        .history-table th {

            font-size: 11px;

            text-transform:
                uppercase;

            letter-spacing: .5px;

            color: var(--muted);

            background:
                #f9fafb;

            padding:
                14px 18px;
        }


        .history-table td {

            padding:
                15px 18px;

            font-size: 13px;

            vertical-align: middle;
        }


        .article-preview {

            max-width: 350px;

            white-space: nowrap;

            overflow: hidden;

            text-overflow: ellipsis;
        }


        .table-result {

            font-size: 11px;

            font-weight: 800;

            padding:
                6px 9px;

            border-radius: 7px;
        }


        .table-fake {

            background:
                #fef2f2;

            color:
                #b91c1c;
        }


        .table-real {

            background:
                #ecfdf5;

            color:
                #15803d;
        }


        /* ==================================================
           LOADING
        ================================================== */

        .loading-container {

            display: none;

            align-items: center;

            gap: 10px;

            color:
                #4f46e5;

            font-weight: 700;

            font-size: 13px;

            margin-top: 15px;
        }


        /* ==================================================
           FOOTER
        ================================================== */

        footer {

            border-top:
                1px solid var(--border);

            padding:
                25px 0;

            color:
                var(--muted);

            font-size: 12px;
        }


        /* ==================================================
           RESPONSIVE
        ================================================== */

        @media(max-width: 768px) {

            .main-container {

                padding-top: 30px;
            }

            .pipeline {

                justify-content:
                    flex-start;
            }

            .pipeline-arrow {

                display: none;
            }

            .hero-title {

                font-size: 31px;
            }

        }

    </style>

</head>


<body>


<!-- ========================================================
     NAVBAR
========================================================= -->

<nav class="navbar navbar-expand-lg navbar-custom">

    <div class="container-fluid px-4">

        <a
            class="navbar-brand d-flex align-items-center gap-3"
            href="#"
        >

            <div class="brand-icon">

                <i class="bi bi-shield-check"></i>

            </div>

            <div>

                <div class="brand-title">
                    NewsGuard AI
                </div>

                <div class="brand-subtitle">
                    Fake News Detection
                </div>

            </div>

        </a>


        <button
            class="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarNav"
        >

            <span class="navbar-toggler-icon"></span>

        </button>


        <div
            class="collapse navbar-collapse"
            id="navbarNav"
        >

            <ul class="navbar-nav ms-auto align-items-lg-center">

                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="#dashboard"
                    >
                        Dashboard
                    </a>

                </li>


                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="#history"
                    >
                        History
                    </a>

                </li>


                <li class="nav-item">

                    <a
                        class="nav-link"
                        href="#about"
                    >
                        About
                    </a>

                </li>


                <li class="nav-item ms-lg-3 mt-2 mt-lg-0">

                    <span
                        class="d-flex align-items-center gap-2
                        small fw-bold text-success"
                    >

                        <span class="online-dot"></span>

                        AI Online

                    </span>

                </li>

            </ul>

        </div>

    </div>

</nav>


<!-- ========================================================
     MAIN
========================================================= -->

<main
    class="main-container"
    id="dashboard"
>


    <!-- HERO -->

    <div class="d-flex flex-wrap
                justify-content-between
                align-items-end
                gap-4
                mb-4">

        <div>

            <h1 class="hero-title">
                AI-Powered Fake News Detection
            </h1>

            <p class="hero-description mb-0">

                Analyze news using BERT,
                Natural Language Processing,
                and knowledge-guided linguistic analysis.

            </p>

        </div>


        <div class="model-pill">

            <i class="bi bi-cpu"></i>

            BERT + NLP + Knowledge

        </div>

    </div>


    <!-- ====================================================
         STATISTICS
    ===================================================== -->

    <div class="row g-3 mb-4">


        <div class="col-6 col-lg-3">

            <div class="stat-card">

                <div class="stat-icon icon-blue">

                    <i class="bi bi-newspaper"></i>

                </div>

                <div class="stat-label">
                    Articles
                </div>

                <div
                    class="stat-value"
                    id="totalArticles"
                >
                    0
                </div>

            </div>

        </div>


        <div class="col-6 col-lg-3">

            <div class="stat-card">

                <div class="stat-icon icon-green">

                    <i class="bi bi-check-circle"></i>

                </div>

                <div class="stat-label">
                    Real
                </div>

                <div
                    class="stat-value text-success"
                    id="realArticles"
                >
                    0
                </div>

            </div>

        </div>


        <div class="col-6 col-lg-3">

            <div class="stat-card">

                <div class="stat-icon icon-red">

                    <i class="bi bi-exclamation-triangle"></i>

                </div>

                <div class="stat-label">
                    Fake
                </div>

                <div
                    class="stat-value text-danger"
                    id="fakeArticles"
                >
                    0
                </div>

            </div>

        </div>


        <div class="col-6 col-lg-3">

            <div class="stat-card">

                <div class="stat-icon icon-orange">

                    <i class="bi bi-graph-up"></i>

                </div>

                <div class="stat-label">
                    Model Accuracy
                </div>

                <div
                    class="stat-value"
                    id="modelAccuracy"
                >
                    Loading...
                </div>

            </div>

        </div>

    </div>


    <!-- ====================================================
         ANALYZER + MODEL STATUS
    ===================================================== -->

    <div
        class="row g-4 mb-4"
        id="analyzer"
    >


        <!-- NEWS ANALYZER -->

        <div class="col-lg-8">

            <div class="dashboard-card">

                <div class="card-header-custom">

                    <div class="card-title-custom">

                        <i class="bi bi-file-earmark-text
                                  me-2 text-primary"></i>

                        News Analyzer

                    </div>

                    <div class="card-subtitle-custom">

                        Paste an article or upload a
                        .txt file for analysis.

                    </div>

                </div>


                <div class="card-body-custom">


                    <textarea
                        id="newsText"
                        class="news-textarea"
                        placeholder="Paste your news article here for analysis..."
                    ></textarea>


                    <div
                        class="d-flex flex-wrap
                               justify-content-between
                               align-items-center
                               gap-3
                               mt-3"
                    >


                        <div>

                            <label
                                class="btn btn-secondary-custom"
                            >

                                <i
                                    class="bi bi-upload me-1"
                                ></i>

                                Upload .txt

                                <input
                                    type="file"
                                    id="fileInput"
                                    accept=".txt"
                                    hidden
                                >

                            </label>

                        </div>


                        <div class="d-flex gap-2">

                            <button
                                class="btn btn-secondary-custom"
                                onclick="clearText()"
                            >

                                <i
                                    class="bi bi-x-circle me-1"
                                ></i>

                                Clear

                            </button>


                            <button
                                class="btn btn-primary-custom"
                                id="analyzeBtn"
                                onclick="analyzeNews()"
                            >

                                <i
                                    class="bi bi-stars me-1"
                                ></i>

                                Analyze News

                                <i
                                    class="bi bi-arrow-right ms-1"
                                ></i>

                            </button>

                        </div>

                    </div>


                    <div
                        class="loading-container"
                        id="loading"
                    >

                        <div
                            class="spinner-border
                                   spinner-border-sm"
                        ></div>

                        Analyzing article with BERT...

                    </div>

                </div>

            </div>

        </div>


        <!-- MODEL STATUS -->

        <div class="col-lg-4">

            <div class="dashboard-card">

                <div class="card-header-custom">

                    <div class="card-title-custom">

                        <i
                            class="bi bi-cpu me-2 text-primary"
                        ></i>

                        Model Status

                    </div>

                    <div class="card-subtitle-custom">

                        Current AI configuration

                    </div>

                </div>


                <div class="card-body-custom">


                    <div class="status-box">

                        <div class="status-indicator"></div>

                        <div>

                            <div class="status-title">
                                BERT Model Ready
                            </div>

                            <div class="status-description">
                                System is ready for prediction
                            </div>

                        </div>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Dataset
                        </span>

                        <span class="info-value">
                            WELFake
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Dataset Size
                        </span>

                        <span class="info-value">
                            72,134
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Model
                        </span>

                        <span class="info-value">
                            BERT
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Knowledge
                        </span>

                        <span class="info-value text-success">
                            Enabled
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Training
                        </span>

                        <span class="info-value">
                            Progressive
                        </span>

                    </div>


                    <div class="info-row">

                        <span class="info-label">
                            Test Accuracy
                        </span>

                        <span
                            class="info-value text-success"
                            id="modelAccuracyStatus"
                        >
                            Loading...
                        </span>

                    </div>


                </div>

            </div>

        </div>

    </div>


    <!-- ====================================================
         RESULT + CONFIDENCE
    ===================================================== -->

    <div
        class="row g-4 mb-4"
        id="resultSection"
    >


        <!-- RESULT -->

        <div class="col-lg-7">

            <div class="dashboard-card">

                <div class="card-header-custom">

                    <div class="card-title-custom">

                        <i
                            class="bi bi-clipboard2-data
                                   me-2 text-primary"
                        ></i>

                        Analysis Result

                    </div>

                    <div class="card-subtitle-custom">

                        Classification and linguistic analysis

                    </div>

                </div>


                <div
                    class="card-body-custom"
                    id="resultArea"
                >

                    <div class="result-placeholder">

                        <div class="result-icon">

                            <i class="bi bi-search"></i>

                        </div>

                        <h6 class="fw-bold">
                            No analysis yet
                        </h6>

                        <p class="small mb-0">
                            Enter news content above and
                            click Analyze News.
                        </p>

                    </div>

                </div>

            </div>

        </div>


        <!-- CONFIDENCE -->

        <div class="col-lg-5">

            <div class="dashboard-card">

                <div class="card-header-custom">

                    <div class="card-title-custom">

                        <i
                            class="bi bi-speedometer2
                                   me-2 text-primary"
                        ></i>

                        AI Confidence

                    </div>

                    <div class="card-subtitle-custom">

                        Prediction probability

                    </div>

                </div>


                <div
                    class="card-body-custom text-center"
                    id="confidenceArea"
                >

                    <div class="result-placeholder">

                        <div class="result-icon">

                            <i class="bi bi-bar-chart"></i>

                        </div>

                        <p class="small mb-0">
                            Confidence will appear here
                            after analysis.
                        </p>

                    </div>

                </div>

            </div>

        </div>

    </div>


    <!-- ====================================================
         DETECTION PIPELINE
    ===================================================== -->

    <div class="dashboard-card mb-4">

        <div class="card-header-custom">

            <div class="card-title-custom">

                <i
                    class="bi bi-diagram-3
                           me-2 text-primary"
                ></i>

                Detection Pipeline

            </div>

            <div class="card-subtitle-custom">

                Processing workflow used by the system

            </div>

        </div>


        <div class="card-body-custom">

            <div class="pipeline">


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        01
                    </div>

                    <div class="pipeline-title">
                        Input News
                    </div>

                </div>


                <div class="pipeline-arrow">
                    →
                </div>


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        02
                    </div>

                    <div class="pipeline-title">
                        NLP
                    </div>

                </div>


                <div class="pipeline-arrow">
                    →
                </div>


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        03
                    </div>

                    <div class="pipeline-title">
                        Tokenization
                    </div>

                </div>


                <div class="pipeline-arrow">
                    →
                </div>


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        04
                    </div>

                    <div class="pipeline-title">
                        BERT
                    </div>

                </div>


                <div class="pipeline-arrow">
                    →
                </div>


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        05
                    </div>

                    <div class="pipeline-title">
                        Knowledge
                    </div>

                </div>


                <div class="pipeline-arrow">
                    →
                </div>


                <div class="pipeline-step">

                    <div class="pipeline-number">
                        06
                    </div>

                    <div class="pipeline-title">
                        Prediction
                    </div>

                </div>


            </div>

        </div>

    </div>


    <!-- ====================================================
         HISTORY
    ===================================================== -->

    <div
        class="dashboard-card mb-4"
        id="history"
    >

        <div
            class="card-header-custom
                   d-flex
                   justify-content-between
                   align-items-center
                   flex-wrap
                   gap-3"
        >

            <div>

                <div class="card-title-custom">

                    <i
                        class="bi bi-clock-history
                               me-2 text-primary"
                    ></i>

                    Recent Predictions

                </div>

                <div class="card-subtitle-custom">

                    Latest analyzed news articles

                </div>

            </div>


            <button
                class="btn btn-secondary-custom"
                onclick="clearHistory()"
            >

                <i
                    class="bi bi-trash3 me-1"
                ></i>

                Clear History

            </button>

        </div>


        <div class="table-wrapper">

            <table
                class="table history-table"
            >

                <thead>

                    <tr>

                        <th>
                            Date
                        </th>

                        <th>
                            Article
                        </th>

                        <th>
                            Result
                        </th>

                        <th>
                            Confidence
                        </th>

                    </tr>

                </thead>


                <tbody id="historyBody">

                </tbody>

            </table>

        </div>

    </div>


    <!-- ====================================================
         ABOUT
    ===================================================== -->

    <div
        class="dashboard-card mb-4"
        id="about"
    >

        <div class="card-header-custom">

            <div class="card-title-custom">

                <i
                    class="bi bi-info-circle
                           me-2 text-primary"
                ></i>

                About NewsGuard AI

            </div>

        </div>


        <div class="card-body-custom">

            <p class="text-secondary mb-2">

                NewsGuard AI is a research prototype
                for fake-news classification using
                Natural Language Processing and
                BERT-based text classification.

            </p>

            <p class="text-secondary mb-0">

                The system combines BERT predictions
                with knowledge-guided linguistic signals
                to provide additional supporting analysis.

            </p>

        </div>

    </div>

</main>


<!-- ========================================================
     FOOTER
========================================================= -->

<footer>

    <div class="container text-center">

        <div class="fw-bold mb-1">
            NewsGuard AI
        </div>

        <div>
            Fake News Detection Using NLP |
            BERT-based Research Prototype
        </div>

    </div>

</footer>


<!-- ========================================================
     JAVASCRIPT
========================================================= -->

<script
    src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
></script>


<script>


// ==========================================================
// FILE UPLOAD
// ==========================================================

document
    .getElementById("fileInput")
    .addEventListener(
        "change",
        function(event) {

            const file =
                event.target.files[0];

            if (!file) {
                return;
            }

            const reader =
                new FileReader();

            reader.onload =
                function(e) {

                    document
                        .getElementById(
                            "newsText"
                        )
                        .value =
                        e.target.result;

                };

            reader.readAsText(file);
        }
    );


// ==========================================================
// CLEAR TEXT
// ==========================================================

function clearText() {

    document
        .getElementById(
            "newsText"
        )
        .value = "";

    document
        .getElementById(
            "fileInput"
        )
        .value = "";

}


// ==========================================================
// ANALYZE NEWS
// ==========================================================

async function analyzeNews() {

    const text =
        document
            .getElementById(
                "newsText"
            )
            .value
            .trim();


    if (!text) {

        alert(
            "Please enter or upload a news article."
        );

        return;
    }


    const button =
        document
            .getElementById(
                "analyzeBtn"
            );


    const loading =
        document
            .getElementById(
                "loading"
            );


    button.disabled = true;

    loading.style.display =
        "flex";


    try {

        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            text: text
                        })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        displayResult(data);

        loadHistory();

        loadStatistics();


        document
            .getElementById(
                "resultSection"
            )
            .scrollIntoView({
                behavior: "smooth"
            });


    }
    catch(error) {

        alert(
            "Error: " +
            error.message
        );

    }
    finally {

        button.disabled = false;

        loading.style.display =
            "none";

    }

}


// ==========================================================
// DISPLAY RESULT
// ==========================================================

function displayResult(data) {

    const prediction =
        String(
            data.prediction || ""
        );


    const confidence =
        Number(
            data.confidence || 0
        );


    const isFake =
        prediction.toLowerCase()
            .includes("fake");


    const resultArea =
        document
            .getElementById(
                "resultArea"
            );


    const confidenceArea =
        document
            .getElementById(
                "confidenceArea"
            );


    // ------------------------------------------------------
    // SEPARATE WARNING LAYER
    // ------------------------------------------------------

    const warningLayer =
        data.warning_layer || data.explanation || {};

    const warningItems =
        Array.isArray(warningLayer.items)
            ? warningLayer.items
            : [];

    const warningStatus =
        String(
            warningLayer.status ||
            (warningItems.length
                ? "Warning patterns found"
                : "No listed warning patterns found")
        );

    const warningBorder =
        warningItems.length ? "#f59e0b" : "#9ca3af";

    let signalsHTML = `
        <div class="mt-4 p-3 rounded-3"
             style="border:1px solid #e5e7eb;border-left:4px solid ${warningBorder};background:#fff;">
            <div class="d-flex align-items-start gap-2 mb-2">
                <i class="bi bi-shield-exclamation text-warning fs-5"></i>
                <div>
                    <div class="fw-bold">Separate Warning Layer</div>
                    <div class="small text-secondary">${escapeHtml(warningStatus)}</div>
                </div>
            </div>
    `;

    if (warningItems.length) {
        signalsHTML += `<div class="mt-2">`;

        warningItems.forEach(function(item) {
            signalsHTML += `
                <span class="signal-badge"
                      title="Matched phrase: ${escapeHtml(item.phrase || "")}">
                    ${escapeHtml(item.category || "Warning signal")}
                    <span class="text-secondary">· ${escapeHtml(item.phrase || "")}</span>
                </span>
            `;
        });

        signalsHTML += `</div>`;
    } else {
        signalsHTML += `
            <div class="small text-secondary mt-2">
                No patterns from the current warning rules were found. This does not mean the article is true.
            </div>
        `;
    }

    signalsHTML += `
            <div class="small text-secondary mt-3">
                ${escapeHtml(warningLayer.message || "Warning signals are not proof of factual accuracy.")}
            </div>
        </div>
    `;


    // ------------------------------------------------------
    // RESULT CARD
    // ------------------------------------------------------

    resultArea.innerHTML =
        `
        <div>

            <div
                class="prediction-badge
                ${
                    isFake
                        ? "prediction-fake"
                        : "prediction-real"
                }"
            >

                <i
                    class="bi
                    ${
                        isFake
                            ? "bi-exclamation-triangle"
                            : "bi-check-circle"
                    }"
                ></i>

                ${
                    isFake
                        ? "POSSIBLY FAKE"
                        : "POSSIBLY REAL"
                }

            </div>


            <div class="mt-4">

                <div
                    class="d-flex
                           justify-content-between
                           mb-2"
                >

                    <span
                        class="small fw-bold"
                    >
                        Model Confidence
                    </span>

                    <span
                        class="small fw-bold"
                    >
                        ${confidence.toFixed(1)}%
                    </span>

                </div>


                <div class="progress">

                    <div
                        class="progress-bar
                        ${
                            isFake
                                ? "bg-danger"
                                : "bg-success"
                        }"
                        style="
                            width:
                            ${confidence}%
                        "
                    ></div>

                </div>

            </div>


            <div
                class="mt-4
                       p-3
                       rounded-3
                       bg-light"
            >

                <div
                    class="small text-secondary"
                >
                    Final Prediction
                </div>

                <div
                    class="fs-4 fw-bold
                    ${
                        isFake
                            ? "text-danger"
                            : "text-success"
                    }"
                >
                    ${escapeHtml(
                        prediction.toUpperCase()
                    )}
                </div>

            </div>


            ${signalsHTML}

        </div>
        `;


    // ------------------------------------------------------
    // CONFIDENCE CARD
    // ------------------------------------------------------

    confidenceArea.innerHTML =
        `
        <div
            class="d-flex
                   flex-column
                   align-items-center
                   justify-content-center"
            style="min-height:300px"
        >

            <div
                class="confidence-number
                ${
                    isFake
                        ? "text-danger"
                        : "text-success"
                }"
            >
                ${confidence.toFixed(1)}%
            </div>

            <div
                class="confidence-label mb-4"
            >
                AI Model Confidence
            </div>


            <div
                class="progress w-100"
                style="max-width:320px;height:14px"
            >

                <div
                    class="progress-bar
                    ${
                        isFake
                            ? "bg-danger"
                            : "bg-success"
                    }"
                    style="
                        width:
                        ${confidence}%
                    "
                ></div>

            </div>


            <div class="mt-4">

                <span
                    class="prediction-badge
                    ${
                        isFake
                            ? "prediction-fake"
                            : "prediction-real"
                    }"
                >

                    <i
                        class="bi
                        ${
                            isFake
                                ? "bi-x-circle"
                                : "bi-check-circle"
                        }"
                    ></i>

                    Prediction:
                    ${escapeHtml(
                        prediction.toUpperCase()
                    )}

                </span>

            </div>

        </div>
        `;

}


// ==========================================================
// LOAD HISTORY
// ==========================================================

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/history"
            );


        const data =
            await response.json();


        const tbody =
            document
                .getElementById(
                    "historyBody"
                );


        tbody.innerHTML = "";


        if (
            !data.history ||
            data.history.length === 0
        ) {

            tbody.innerHTML =
                `
                <tr>

                    <td
                        colspan="4"
                        class="text-center
                               text-secondary
                               py-5"
                    >

                        No predictions yet.

                    </td>

                </tr>
                `;

            return;
        }


        data.history.forEach(
            function(item) {

                const isFake =
                    String(
                        item.prediction
                    )
                    .toLowerCase()
                    .includes("fake");


                const row =
                    document.createElement(
                        "tr"
                    );


                row.innerHTML =
                    `
                    <td>
                        ${escapeHtml(
                            item.created_at
                        )}
                    </td>

                    <td>

                        <div
                            class="article-preview"
                            title="${escapeHtml(
                                item.news_text
                            )}"
                        >
                            ${escapeHtml(
                                item.news_text
                            )}
                        </div>

                    </td>

                    <td>

                        <span
                            class="table-result
                            ${
                                isFake
                                    ? "table-fake"
                                    : "table-real"
                            }"
                        >

                            ${
                                isFake
                                    ? "FAKE"
                                    : "REAL"
                            }

                        </span>

                    </td>

                    <td>

                        <strong>
                            ${Number(
                                item.confidence
                            ).toFixed(1)}%
                        </strong>

                    </td>
                    `;


                tbody.appendChild(row);

            }
        );

    }
    catch(error) {

        console.error(
            "History error:",
            error
        );

    }

}


// ==========================================================
// LOAD STATISTICS
// ==========================================================

async function loadStatistics() {

    try {

        const response = await fetch("/statistics");
        const data = await response.json();

        document.getElementById("totalArticles").textContent = data.total || 0;
        document.getElementById("realArticles").textContent = data.real || 0;
        document.getElementById("fakeArticles").textContent = data.fake || 0;
        if (data.model_accuracy !== null && data.model_accuracy !== undefined) {

            const accuracyText =
                Number(data.model_accuracy).toFixed(2) + "%";

            document.getElementById("modelAccuracy").textContent =
                accuracyText;

            const statusAccuracy =
                document.getElementById("modelAccuracyStatus");

            if (statusAccuracy) {
                statusAccuracy.textContent =
                    accuracyText;
            }

        } else {

            document.getElementById("modelAccuracy").textContent =
                "Unavailable";

            const statusAccuracy =
                document.getElementById("modelAccuracyStatus");

            if (statusAccuracy) {
                statusAccuracy.textContent =
                    "Unavailable";
            }
        }

    } catch(error) {
        console.error("Statistics error:", error);
    }
}


// ==========================================================
// CLEAR HISTORY
// ==========================================================

async function clearHistory() {

    const confirmed =
        confirm(
            "Are you sure you want to clear prediction history?"
        );


    if (!confirmed) {
        return;
    }


    try {

        const response =
            await fetch(
                "/clear-history",
                {
                    method: "POST"
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Unable to clear history."
            );

        }


        loadHistory();

        loadStatistics();

        alert(
            "Prediction history cleared successfully."
        );

    }
    catch(error) {

        alert(
            "Error: " +
            error.message
        );

    }

}


// ==========================================================
// HTML ESCAPE
// ==========================================================

function escapeHtml(value) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent =
        value == null
            ? ""
            : String(value);

    return div.innerHTML;
}


// ==========================================================
// INITIAL LOAD
// ==========================================================

document.addEventListener(
    "DOMContentLoaded",
    function() {

        loadHistory();

        loadStatistics();

    }
);

</script>


</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )


        if not data:

            return jsonify({
                "error":
                    "No JSON data received."
            }), 400


        text = str(
            data.get(
                "text",
                ""
            )
        ).strip()


        if not text:

            return jsonify({
                "error":
                    "News text is empty."
            }), 400


        print()
        print("=" * 60)
        print("NEW PREDICTION REQUEST")
        print("=" * 60)

        print(
            "Text length:",
            len(text)
        )


        # ----------------------------------------------------
        # BERT PREDICTION
        # ----------------------------------------------------

        prediction, confidence = \
            predict_news(text)


        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            confidence
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            confidence
        )


        if 0 <= confidence <= 1:

            confidence *= 100


        confidence = max(
            0,
            min(
                100,
                round(
                    confidence,
                    2
                )
            )
        )


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        warning_layer = get_warning_layer(text)
        explanation = warning_layer  # Keep compatibility with the existing UI.


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        save_prediction(
            text,
            str(prediction),
            confidence,
            warning_layer["message"],
        )


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return jsonify({

            "prediction":
                str(prediction),

            "confidence":
                confidence,

            "explanation": explanation,

            "warning_layer": warning_layer,

        })


    except Exception as e:

        print()
        print("=" * 60)
        print("PREDICTION ERROR")
        print("=" * 60)

        print(
            type(e).__name__,
            ":",
            str(e)
        )

        print("=" * 60)


        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# HISTORY API
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    try:

        return jsonify({

            "history":
                get_history()

        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# STATISTICS API
# ============================================================

@app.route(
    "/statistics",
    methods=["GET"]
)
def statistics():

    try:
        stats = get_statistics()
        model_accuracy = get_model_accuracy()

        return jsonify({
            "total": stats["total"],
            "real": stats["real"],
            "fake": stats["fake"],
            "model_accuracy": model_accuracy
        })

    except Exception as e:
        print("Statistics error:", e)

        return jsonify({
            "error": str(e),
            "model_accuracy": None
        }), 500


# ============================================================
# CLEAR HISTORY API
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def delete_history():

    try:

        clear_history()


        return jsonify({

            "message":
                "Prediction history cleared successfully."

        })


    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# START APPLICATION
# ============================================================

init_database()


if __name__ == "__main__":

    print()
    print("=" * 65)
    print("        NEWSGUARD AI - FAKE NEWS DETECTION")
    print("=" * 65)
    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print()
    print("Model:")
    print("BERT + NLP + Knowledge Guided Analysis")
    print()
    print("Database:")
    print("prediction_history.db")
    print()
    print("=" * 65)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

