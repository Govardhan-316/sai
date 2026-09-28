
from flask import Flask, request, jsonify, render_template_string
import sqlite3
from datetime import datetime

from predict import predict_news, analyze_suspicious_language


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

DATABASE = "prediction_history.db"


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
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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


def clear_history():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions"
    )

    conn.commit()
    conn.close()


# ============================================================
# SUSPICIOUS LANGUAGE ANALYSIS
# ============================================================

def get_explanation(text):

    try:

        suspicious = analyze_suspicious_language(text)

    except Exception as e:

        print(
            "Suspicious language error:",
            e
        )

        suspicious = []


    if not suspicious:

        return {
            "count": 0,
            "items": [],
            "message":
                "No strong suspicious linguistic patterns were detected."
        }


    items = []


    for item in suspicious:

        if isinstance(item, dict):

            category = item.get(
                "category",
                "Suspicious Language"
            )

            phrases = item.get(
                "phrases",
                []
            )


            if isinstance(phrases, list):

                for phrase in phrases:

                    items.append({
                        "category": category,
                        "phrase": str(phrase)
                    })


        else:

            items.append({
                "category":
                    "Suspicious Language",

                "phrase":
                    str(item)
            })


    return {
        "count": len(items),

        "items": items,

        "message":
            "Suspicious linguistic patterns were detected. "
            "These are supporting linguistic signals and do not "
            "independently verify the factual truth of the article."
    }


# ============================================================
# HTML DASHBOARD
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
    Fake News Detection Using NLP
</title>


<style>

/* ============================================================
   GENERAL
   ============================================================ */

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f7fb;

    color: #172033;

    min-height: 100vh;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

.sidebar {

    position: fixed;

    left: 0;
    top: 0;
    bottom: 0;

    width: 240px;

    background: #101828;

    color: white;

    padding: 25px 18px;

    z-index: 10;
}


.logo {

    display: flex;

    align-items: center;

    gap: 12px;

    padding: 0 10px 25px;

    border-bottom:
        1px solid
        rgba(255,255,255,0.08);

    margin-bottom: 25px;
}


.logo-icon {

    width: 42px;
    height: 42px;

    border-radius: 10px;

    background: #2563eb;

    display: flex;

    align-items: center;

    justify-content: center;

    font-weight: bold;
}


.logo-text {

    font-size: 16px;

    font-weight: bold;
}


.logo-subtitle {

    color: #98a2b3;

    font-size: 11px;

    margin-top: 3px;
}


.nav-title {

    color: #667085;

    font-size: 10px;

    font-weight: bold;

    text-transform: uppercase;

    padding: 0 12px;

    margin-bottom: 10px;
}


.nav-item {

    display: flex;

    align-items: center;

    gap: 12px;

    padding: 12px;

    border-radius: 8px;

    margin-bottom: 5px;

    color: #c9d1dc;

    font-size: 14px;

    cursor: pointer;
}


.nav-item:hover {

    background: #1d2939;

    color: white;
}


.nav-item.active {

    background: #1d4ed8;

    color: white;
}


.nav-icon {

    width: 25px;

    text-align: center;

    font-weight: bold;
}


.sidebar-bottom {

    position: absolute;

    bottom: 25px;

    left: 18px;

    right: 18px;

    background: #172033;

    padding: 14px;

    border-radius: 10px;
}


.status-row {

    display: flex;

    align-items: center;

    gap: 8px;

    font-size: 12px;
}


.status-dot {

    width: 8px;
    height: 8px;

    background: #12b76a;

    border-radius: 50%;
}


.status-text {

    color: #98a2b3;

    font-size: 11px;

    margin-top: 5px;

    padding-left: 16px;
}


/* ============================================================
   MAIN
   ============================================================ */

.main {

    margin-left: 240px;

    min-height: 100vh;
}


.topbar {

    height: 70px;

    background: white;

    border-bottom:
        1px solid #eaecf0;

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0 30px;
}


.breadcrumb {

    color: #667085;

    font-size: 13px;
}


.breadcrumb strong {

    color: #172033;
}


.top-status {

    display: flex;

    align-items: center;

    gap: 8px;

    color: #475467;

    font-size: 13px;
}


.green-dot {

    width: 8px;
    height: 8px;

    background: #12b76a;

    border-radius: 50%;
}


/* ============================================================
   CONTENT
   ============================================================ */

.content {

    padding: 30px;

    max-width: 1500px;

    margin: auto;
}


.page-header {

    display: flex;

    justify-content: space-between;

    align-items: flex-end;

    margin-bottom: 25px;

    gap: 20px;
}


.page-title {

    font-size: 28px;

    margin-bottom: 7px;

    color: #101828;
}


.page-description {

    color: #667085;

    font-size: 14px;

    max-width: 750px;

    line-height: 1.5;
}


.model-badge {

    padding: 8px 13px;

    background: #eff6ff;

    color: #1d4ed8;

    border:
        1px solid #bfdbfe;

    border-radius: 20px;

    font-size: 12px;

    font-weight: bold;

    white-space: nowrap;
}


/* ============================================================
   GRID
   ============================================================ */

.dashboard-grid {

    display: grid;

    grid-template-columns:
        minmax(0, 1.55fr)
        minmax(300px, 0.85fr);

    gap: 22px;

    align-items: start;
}


/* ============================================================
   CARDS
   ============================================================ */

.card {

    background: white;

    border:
        1px solid #eaecf0;

    border-radius: 12px;

    box-shadow:
        0 2px 5px
        rgba(16,24,40,0.03);

    overflow: hidden;
}


.card-header {

    padding: 20px 22px;

    border-bottom:
        1px solid #eaecf0;

    display: flex;

    align-items: center;

    justify-content: space-between;
}


.card-title {

    font-size: 15px;

    font-weight: bold;

    color: #101828;
}


.card-subtitle {

    font-size: 12px;

    color: #667085;

    margin-top: 4px;
}


.card-body {

    padding: 22px;
}


/* ============================================================
   TEXTAREA
   ============================================================ */

textarea {

    width: 100%;

    min-height: 250px;

    resize: vertical;

    border:
        1px solid #d0d5dd;

    border-radius: 9px;

    padding: 16px;

    font-family: Arial, sans-serif;

    font-size: 14px;

    line-height: 1.6;

    color: #344054;

    outline: none;
}


textarea:focus {

    border-color: #2563eb;

    box-shadow:
        0 0 0 3px
        rgba(37,99,235,0.10);
}


/* ============================================================
   BUTTONS
   ============================================================ */

.input-footer {

    margin-top: 13px;

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 15px;
}


.action-buttons {

    display: flex;

    gap: 9px;
}


.primary-btn,
.secondary-btn {

    padding: 11px 18px;

    border-radius: 7px;

    font-size: 13px;

    font-weight: bold;

    cursor: pointer;
}


.primary-btn {

    border: none;

    background: #2563eb;

    color: white;
}


.primary-btn:hover {

    background: #1d4ed8;
}


.primary-btn:disabled {

    background: #98a2b3;

    cursor: not-allowed;
}


.secondary-btn {

    background: #f8fafc;

    border:
        1px solid #d0d5dd;

    color: #344054;
}


.secondary-btn:hover {

    background: #f2f4f7;
}


.file-upload {

    position: relative;

    display: inline-flex;
}


.file-upload input {

    position: absolute;

    opacity: 0;

    width: 100%;

    height: 100%;

    cursor: pointer;
}


/* ============================================================
   LOADING
   ============================================================ */

.loading {

    display: none;

    align-items: center;

    justify-content: center;

    gap: 8px;

    color: #667085;

    font-size: 12px;

    margin-top: 14px;
}


.spinner {

    width: 15px;

    height: 15px;

    border:
        2px solid #d0d5dd;

    border-top-color: #2563eb;

    border-radius: 50%;

    animation:
        spin 0.7s linear infinite;
}


@keyframes spin {

    to {
        transform: rotate(360deg);
    }
}


/* ============================================================
   RESULT
   ============================================================ */

.result-area {

    min-height: 310px;

    display: flex;

    align-items: center;

    justify-content: center;

    text-align: center;

    padding: 25px;
}


.empty-result {

    color: #98a2b3;

    font-size: 13px;
}


.result-content {

    width: 100%;
}


.result-label {

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1px;

    color: #667085;

    font-weight: bold;

    margin-bottom: 10px;
}


.prediction {

    font-size: 36px;

    font-weight: 800;

    margin-bottom: 18px;
}


.prediction.real {

    color: #039855;
}


.prediction.fake {

    color: #d92d20;
}


.confidence-wrapper {

    max-width: 440px;

    margin: auto;
}


.confidence-top {

    display: flex;

    justify-content: space-between;

    margin-bottom: 8px;

    font-size: 12px;

    color: #667085;
}


.confidence-value {

    color: #101828;

    font-weight: bold;
}


.progress {

    width: 100%;

    height: 10px;

    background: #eaecf0;

    border-radius: 20px;

    overflow: hidden;
}


.progress-bar {

    height: 100%;

    background: #2563eb;

    border-radius: 20px;

    transition:
        width 0.6s ease;
}


/* ============================================================
   EXPLANATION
   ============================================================ */

.explanation {

    margin-top: 25px;

    border-top:
        1px solid #eaecf0;

    padding-top: 20px;

    text-align: left;
}


.explanation-title {

    font-size: 13px;

    font-weight: bold;

    margin-bottom: 10px;

    color: #344054;
}


.explanation-message {

    font-size: 12px;

    color: #667085;

    line-height: 1.5;

    margin-bottom: 12px;
}


.keyword-list {

    display: flex;

    flex-wrap: wrap;

    gap: 7px;
}


.keyword {

    display: inline-flex;

    flex-direction: column;

    gap: 2px;

    padding: 7px 10px;

    background: #fff7ed;

    border:
        1px solid #fed7aa;

    border-radius: 7px;
}


.keyword-category {

    font-size: 9px;

    color: #9a3412;

    font-weight: bold;

    text-transform: uppercase;
}


.keyword-word {

    font-size: 11px;

    color: #7c2d12;
}


/* ============================================================
   COMPONENTS
   ============================================================ */

.components-card {

    margin-top: 22px;
}


.component-grid {

    display: grid;

    grid-template-columns:
        repeat(4, minmax(0, 1fr));

    gap: 12px;
}


.component-btn {

    border:
        1px solid #eaecf0;

    background: #f8fafc;

    border-radius: 9px;

    padding: 15px;

    text-align: left;

    cursor: pointer;

    min-height: 112px;
}


.component-btn:hover {

    border-color: #93c5fd;

    background: #eff6ff;
}


.component-btn.active {

    border-color: #2563eb;

    background: #eff6ff;
}


.component-number {

    color: #2563eb;

    font-size: 11px;

    font-weight: bold;

    margin-bottom: 9px;
}


.component-name {

    color: #101828;

    font-size: 13px;

    font-weight: bold;

    margin-bottom: 5px;
}


.component-short {

    color: #667085;

    font-size: 11px;

    line-height: 1.35;
}


.component-details {

    margin-top: 15px;

    padding: 17px;

    background: #f8fafc;

    border:
        1px solid #eaecf0;

    border-radius: 9px;

    display: none;
}


.component-details.show {

    display: block;
}


.details-title {

    color: #101828;

    font-size: 14px;

    font-weight: bold;

    margin-bottom: 7px;
}


.details-text {

    color: #667085;

    font-size: 12px;

    line-height: 1.6;
}


/* ============================================================
   RIGHT SIDE
   ============================================================ */

.right-stack {

    display: flex;

    flex-direction: column;

    gap: 22px;
}


.info-list {

    display: flex;

    flex-direction: column;

    gap: 13px;
}


.info-row {

    display: flex;

    align-items: flex-start;

    justify-content: space-between;

    gap: 15px;

    padding-bottom: 12px;

    border-bottom:
        1px solid #f2f4f7;
}


.info-row:last-child {

    border-bottom: none;

    padding-bottom: 0;
}


.info-label {

    color: #667085;

    font-size: 12px;
}


.info-value {

    color: #344054;

    font-size: 12px;

    font-weight: bold;

    text-align: right;
}


.stat-card {

    padding: 20px;
}


.stat-label {

    color: #667085;

    font-size: 12px;

    margin-bottom: 7px;
}


.stat-value {

    font-size: 26px;

    font-weight: 800;

    color: #101828;
}


.stat-description {

    color: #98a2b3;

    font-size: 11px;

    margin-top: 5px;
}


/* ============================================================
   HISTORY
   ============================================================ */

.history-card {

    margin-top: 22px;
}


.history-table-wrapper {

    overflow-x: auto;
}


table {

    width: 100%;

    border-collapse: collapse;
}


th {

    background: #f8fafc;

    color: #667085;

    font-size: 10px;

    text-transform: uppercase;

    padding: 12px;

    text-align: left;

    white-space: nowrap;
}


td {

    border-top:
        1px solid #eaecf0;

    padding: 13px 12px;

    color: #475467;

    font-size: 12px;

    vertical-align: top;
}


.news-preview {

    max-width: 400px;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}


.history-real {

    color: #039855;

    font-weight: bold;
}


.history-fake {

    color: #d92d20;

    font-weight: bold;
}


.clear-history-btn {

    border:
        1px solid #f04438;

    background: white;

    color: #d92d20;

    padding: 7px 12px;

    border-radius: 6px;

    cursor: pointer;

    font-size: 11px;

    font-weight: bold;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {

    text-align: center;

    color: #98a2b3;

    font-size: 11px;

    margin-top: 35px;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 1100px) {

    .dashboard-grid {

        grid-template-columns: 1fr;
    }

    .component-grid {

        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }
}


@media (max-width: 750px) {

    .sidebar {

        width: 70px;

        padding: 20px 10px;
    }

    .logo-text,
    .logo-subtitle,
    .nav-title,
    .nav-item span,
    .sidebar-bottom {

        display: none;
    }

    .main {

        margin-left: 70px;
    }

    .content {

        padding: 20px 15px;
    }

    .component-grid {

        grid-template-columns: 1fr;
    }

    .page-header {

        flex-direction: column;

        align-items: flex-start;
    }

    .input-footer {

        flex-direction: column;

        align-items: stretch;
    }

    .action-buttons {

        width: 100%;
    }

    .action-buttons button {

        flex: 1;
    }
}

</style>

</head>


<body>


<!-- ============================================================
     SIDEBAR
     ============================================================ -->

<aside class="sidebar">


    <div class="logo">

        <div class="logo-icon">
            AI
        </div>

        <div>

            <div class="logo-text">
                NewsGuard NLP
            </div>

            <div class="logo-subtitle">
                Fake News Detection
            </div>

        </div>

    </div>


    <div class="nav-title">
        Dashboard
    </div>


    <div
        class="nav-item active"
        onclick="scrollToSection('analyzer')"
    >

        <div class="nav-icon">
            01
        </div>

        <span>
            News Analyzer
        </span>

    </div>


    <div
        class="nav-item"
        onclick="scrollToSection('components')"
    >

        <div class="nav-icon">
            02
        </div>

        <span>
            System Components
        </span>

    </div>


    <div
        class="nav-item"
        onclick="scrollToSection('history')"
    >

        <div class="nav-icon">
            03
        </div>

        <span>
            Prediction History
        </span>

    </div>


    <div
        class="nav-item"
        onclick="scrollToSection('model')"
    >

        <div class="nav-icon">
            04
        </div>

        <span>
            Model Information
        </span>

    </div>


    <div class="sidebar-bottom">

        <div class="status-row">

            <div class="status-dot"></div>

            System Online

        </div>

        <div class="status-text">

            BERT model loaded

        </div>

    </div>


</aside>


<!-- ============================================================
     MAIN
     ============================================================ -->

<main class="main">


<header class="topbar">

    <div class="breadcrumb">

        Project /
        <strong>Fake News Detection</strong>

    </div>


    <div class="top-status">

        <div class="green-dot"></div>

        AI model ready

    </div>

</header>


<div class="content">


    <!-- ========================================================
         PAGE HEADER
         ======================================================== -->

    <div class="page-header">

        <div>

            <h1 class="page-title">

                Fake News Detection Dashboard

            </h1>


            <p class="page-description">

                Analyze news content using Natural Language
                Processing, BERT-based classification,
                knowledge-guided linguistic analysis and
                confidence scoring.

            </p>

        </div>


        <div class="model-badge">

            BERT + Knowledge + Progressive Training

        </div>

    </div>


    <!-- ========================================================
         DASHBOARD GRID
         ======================================================== -->

    <div class="dashboard-grid">


        <!-- ====================================================
             LEFT COLUMN
             ==================================================== -->

        <div>


            <!-- NEWS ANALYZER -->

            <section
                class="card"
                id="analyzer"
            >


                <div class="card-header">

                    <div>

                        <div class="card-title">

                            News Content Analyzer

                        </div>

                        <div class="card-subtitle">

                            Paste article content or upload a
                            .txt file

                        </div>

                    </div>

                </div>


                <div class="card-body">


                    <textarea
                        id="newsText"
                        placeholder="Paste the news article here for analysis..."
                    ></textarea>


                    <div class="input-footer">


                        <div class="file-upload">

                            <button
                                class="secondary-btn"
                                type="button"
                            >
                                Upload .txt File
                            </button>


                            <input
                                type="file"
                                id="fileInput"
                                accept=".txt"
                            >

                        </div>


                        <div class="action-buttons">


                            <button
                                class="secondary-btn"
                                type="button"
                                onclick="clearText()"
                            >

                                Clear

                            </button>


                            <button
                                class="primary-btn"
                                id="analyzeBtn"
                                type="button"
                                onclick="analyzeNews()"
                            >

                                Analyze News

                            </button>


                        </div>

                    </div>


                    <div
                        class="loading"
                        id="loading"
                    >

                        <div class="spinner"></div>

                        Analyzing article with BERT...

                    </div>


                </div>

            </section>


            <!-- =================================================
                 RESULT
                 ================================================= -->

            <section
                class="card"
                style="margin-top:22px;"
            >


                <div class="card-header">

                    <div>

                        <div class="card-title">

                            Analysis Result

                        </div>

                        <div class="card-subtitle">

                            Classification and confidence analysis

                        </div>

                    </div>

                </div>


                <div
                    class="result-area"
                    id="resultArea"
                >

                    <div class="empty-result">

                        Enter news content above and click
                        <strong>Analyze News</strong>.

                    </div>

                </div>


            </section>


            <!-- =================================================
                 COMPONENTS
                 ================================================= -->

            <section
                class="card components-card"
                id="components"
            >


                <div class="card-header">

                    <div>

                        <div class="card-title">

                            System Components

                        </div>

                        <div class="card-subtitle">

                            Detection pipeline components

                        </div>

                    </div>

                </div>


                <div class="card-body">


                    <div class="component-grid">


                        <button
                            class="component-btn active"
                            type="button"
                            onclick="showComponent(1, this)"
                        >

                            <div class="component-number">
                                COMPONENT 01
                            </div>

                            <div class="component-name">
                                NLP Preprocessing
                            </div>

                            <div class="component-short">
                                Cleans and prepares news text.
                            </div>

                        </button>


                        <button
                            class="component-btn"
                            type="button"
                            onclick="showComponent(2, this)"
                        >

                            <div class="component-number">
                                COMPONENT 02
                            </div>

                            <div class="component-name">
                                BERT Classification
                            </div>

                            <div class="component-short">
                                Learns contextual language patterns.
                            </div>

                        </button>


                        <button
                            class="component-btn"
                            type="button"
                            onclick="showComponent(3, this)"
                        >

                            <div class="component-number">
                                COMPONENT 03
                            </div>

                            <div class="component-name">
                                Knowledge Analysis
                            </div>

                            <div class="component-short">
                                Detects suspicious language signals.
                            </div>

                        </button>


                        <button
                            class="component-btn"
                            type="button"
                            onclick="showComponent(4, this)"
                        >

                            <div class="component-number">
                                COMPONENT 04
                            </div>

                            <div class="component-name">
                                Confidence Analysis
                            </div>

                            <div class="component-short">
                                Provides prediction probability.
                            </div>

                        </button>


                    </div>


                    <div
                        class="component-details show"
                        id="componentDetails"
                    >

                        <div
                            class="details-title"
                            id="detailsTitle"
                        >
                            NLP Preprocessing
                        </div>


                        <div
                            class="details-text"
                            id="detailsText"
                        >

                            The news article is cleaned and prepared
                            before being passed to the BERT tokenizer
                            and classification model.

                        </div>

                    </div>


                </div>

            </section>


            <!-- =================================================
                 HISTORY
                 ================================================= -->

            <section
                class="card history-card"
                id="history"
            >


                <div class="card-header">

                    <div>

                        <div class="card-title">
                            Prediction History
                        </div>

                        <div class="card-subtitle">
                            Recently analyzed news articles
                        </div>

                    </div>


                    <button
                        class="clear-history-btn"
                        type="button"
                        onclick="clearHistory()"
                    >

                        Clear History

                    </button>

                </div>


                <div class="history-table-wrapper">

                    <table>

                        <thead>

                            <tr>

                                <th>
                                    Date
                                </th>

                                <th>
                                    News
                                </th>

                                <th>
                                    Prediction
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


            </section>


        </div>


        <!-- ====================================================
             RIGHT COLUMN
             ==================================================== -->

        <div class="right-stack">


            <!-- MODEL INFORMATION -->

            <section
                class="card"
                id="model"
            >


                <div class="card-header">

                    <div>

                        <div class="card-title">
                            Model Information
                        </div>

                        <div class="card-subtitle">
                            Current AI configuration
                        </div>

                    </div>

                </div>


                <div class="card-body">


                    <div class="info-list">


                        <div class="info-row">

                            <div class="info-label">
                                Dataset
                            </div>

                            <div class="info-value">
                                WELFake
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                Dataset Size
                            </div>

                            <div class="info-value">
                                72,134
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                Model
                            </div>

                            <div class="info-value">
                                BERT
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                Training
                            </div>

                            <div class="info-value">
                                Progressive
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                Classification
                            </div>

                            <div class="info-value">
                                Real / Fake
                            </div>

                        </div>


                    </div>

                </div>

            </section>


            <!-- ACCURACY -->

            <section class="card stat-card">

                <div class="stat-label">

                    Model Test Accuracy

                </div>


                <div class="stat-value">

                    94.75%

                </div>


                <div class="stat-description">

                    Current experiment

                </div>

            </section>


            <!-- PIPELINE -->

            <section class="card">


                <div class="card-header">

                    <div>

                        <div class="card-title">
                            Detection Pipeline
                        </div>

                        <div class="card-subtitle">
                            Processing workflow
                        </div>

                    </div>

                </div>


                <div class="card-body">


                    <div class="info-list">


                        <div class="info-row">

                            <div class="info-label">
                                01
                            </div>

                            <div class="info-value">
                                Input News
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                02
                            </div>

                            <div class="info-value">
                                Text Processing
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                03
                            </div>

                            <div class="info-value">
                                BERT Tokenization
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                04
                            </div>

                            <div class="info-value">
                                Knowledge Analysis
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                05
                            </div>

                            <div class="info-value">
                                Classification
                            </div>

                        </div>


                        <div class="info-row">

                            <div class="info-label">
                                06
                            </div>

                            <div class="info-value">
                                Confidence Score
                            </div>

                        </div>


                    </div>

                </div>

            </section>


        </div>

    </div>


    <div class="footer">

        Fake News Detection Using NLP |
        BERT-based research prototype

    </div>


</div>

</main>


<script>

/* ============================================================
   FILE UPLOAD
   ============================================================ */

const newsText =
    document.getElementById("newsText");

const fileInput =
    document.getElementById("fileInput");


fileInput.addEventListener(
    "change",
    function () {

        const file = this.files[0];

        if (!file) {
            return;
        }


        if (
            !file.name
                .toLowerCase()
                .endsWith(".txt")
        ) {

            alert(
                "Please select a .txt file."
            );

            this.value = "";

            return;
        }


        const reader =
            new FileReader();


        reader.onload =
            function (event) {

                newsText.value =
                    event.target.result;

            };


        reader.onerror =
            function () {

                alert(
                    "Unable to read the file."
                );

            };


        reader.readAsText(file);

    }
);


/* ============================================================
   ANALYZE NEWS
   ============================================================ */

async function analyzeNews() {

    console.log(
        "Analyze button clicked"
    );


    const text =
        newsText.value.trim();


    const button =
        document.getElementById(
            "analyzeBtn"
        );


    const loading =
        document.getElementById(
            "loading"
        );


    const resultArea =
        document.getElementById(
            "resultArea"
        );


    if (!text) {

        alert(
            "Please enter a news article."
        );

        return;
    }


    button.disabled = true;

    button.innerText =
        "Analyzing...";

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


        console.log(
            "Server response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }


        displayResult(data);

        loadHistory();

    }


    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        resultArea.innerHTML = `

            <div class="empty-result">

                <strong style="color:#d92d20;">
                    Error
                </strong>

                <br><br>

                ${escapeHTML(error.message)}

            </div>

        `;

    }


    finally {

        button.disabled = false;

        button.innerText =
            "Analyze News";

        loading.style.display =
            "none";
    }
}


/* ============================================================
   DISPLAY RESULT
   ============================================================ */

function displayResult(data) {

    const resultArea =
        document.getElementById(
            "resultArea"
        );


    const prediction =
        String(
            data.prediction ||
            "Unknown"
        );


    let confidence =
        Number(
            data.confidence || 0
        );


    if (
        confidence >= 0 &&
        confidence <= 1
    ) {

        confidence =
            confidence * 100;
    }


    confidence =
        Math.max(
            0,
            Math.min(
                100,
                confidence
            )
        );


    const predictionClass =
        prediction
            .toLowerCase()
            .includes("fake")
            ? "fake"
            : "real";


    let keywordsHTML = "";


    if (
        data.explanation &&
        Array.isArray(
            data.explanation.items
        )
    ) {

        data.explanation.items.forEach(
            function (item) {

                keywordsHTML += `

                    <div class="keyword">

                        <span
                            class="keyword-category"
                        >
                            ${escapeHTML(
                                item.category
                            )}
                        </span>

                        <span
                            class="keyword-word"
                        >
                            ${escapeHTML(
                                item.phrase
                            )}
                        </span>

                    </div>

                `;

            }
        );
    }


    let message =
        "No explanation available.";


    if (
        data.explanation &&
        data.explanation.message
    ) {

        message =
            data.explanation.message;
    }


    resultArea.innerHTML = `

        <div class="result-content">


            <div class="result-label">

                Classification Result

            </div>


            <div
                class="prediction ${predictionClass}"
            >

                ${escapeHTML(
                    prediction
                )}

            </div>


            <div
                class="confidence-wrapper"
            >


                <div
                    class="confidence-top"
                >

                    <span>
                        Model Confidence
                    </span>


                    <span
                        class="confidence-value"
                    >

                        ${confidence.toFixed(2)}%

                    </span>

                </div>


                <div class="progress">

                    <div
                        class="progress-bar"
                        style="width:${confidence}%"
                    >
                    </div>

                </div>


            </div>


            <div class="explanation">


                <div
                    class="explanation-title"
                >

                    Linguistic Explanation

                </div>


                <div
                    class="explanation-message"
                >

                    ${escapeHTML(
                        message
                    )}

                </div>


                <div
                    class="keyword-list"
                >

                    ${keywordsHTML}

                </div>


            </div>


        </div>

    `;
}


/* ============================================================
   CLEAR TEXT
   ============================================================ */

function clearText() {

    newsText.value = "";

    fileInput.value = "";


    document.getElementById(
        "resultArea"
    ).innerHTML = `

        <div class="empty-result">

            Enter news content above and click
            <strong>Analyze News</strong>.

        </div>

    `;
}


/* ============================================================
   LOAD HISTORY
   ============================================================ */

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/history"
            );


        const data =
            await response.json();


        const body =
            document.getElementById(
                "historyBody"
            );


        if (
            !data.history ||
            data.history.length === 0
        ) {

            body.innerHTML = `

                <tr>

                    <td
                        colspan="4"
                        style="
                            text-align:center;
                            color:#98a2b3;
                            padding:25px;
                        "
                    >

                        No prediction history available.

                    </td>

                </tr>

            `;

            return;
        }


        body.innerHTML =
            data.history
                .map(
                    function (item) {

                        const isReal =
                            String(
                                item.prediction
                            )
                            .toLowerCase()
                            .includes("real");


                        return `

                            <tr>

                                <td>

                                    ${escapeHTML(
                                        item.created_at
                                    )}

                                </td>


                                <td>

                                    <div
                                        class="news-preview"
                                        title="${escapeHTML(
                                            item.news_text
                                        )}"
                                    >

                                        ${escapeHTML(
                                            item.news_text
                                        )}

                                    </div>

                                </td>


                                <td>

                                    <span
                                        class="${
                                            isReal
                                            ? "history-real"
                                            : "history-fake"
                                        }"
                                    >

                                        ${escapeHTML(
                                            item.prediction
                                        )}

                                    </span>

                                </td>


                                <td>

                                    ${Number(
                                        item.confidence
                                    ).toFixed(2)}%

                                </td>

                            </tr>

                        `;

                    }
                )
                .join("");

    }


    catch (error) {

        console.error(
            "History error:",
            error
        );

    }
}


/* ============================================================
   CLEAR HISTORY
   ============================================================ */

async function clearHistory() {

    const confirmed =
        confirm(
            "Are you sure you want to clear all prediction history?"
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

    }


    catch (error) {

        alert(
            error.message
        );

    }
}


/* ============================================================
   COMPONENT DETAILS
   ============================================================ */

function showComponent(
    number,
    element
) {


    document
        .querySelectorAll(
            ".component-btn"
        )
        .forEach(
            function (button) {

                button.classList.remove(
                    "active"
                );

            }
        );


    element.classList.add(
        "active"
    );


    const title =
        document.getElementById(
            "detailsTitle"
        );


    const text =
        document.getElementById(
            "detailsText"
        );


    const details = {

        1: {

            title:
                "NLP Preprocessing",

            text:
                "The news article is cleaned and prepared before classification. The processed text is passed to the BERT tokenizer and classification model."

        },


        2: {

            title:
                "BERT Classification",

            text:
                "BERT learns contextual relationships between words and sentences. The fine-tuned BERT classifier uses these language patterns to classify the input as Real or Fake."

        },


        3: {

            title:
                "Knowledge-Guided Analysis",

            text:
                "The system checks the article for suspicious linguistic signals such as sensational wording, absolute claims and other suspicious expressions."

        },


        4: {

            title:
                "Confidence Analysis",

            text:
                "The classifier produces a prediction confidence score. The dashboard displays this value as a percentage."

        }

    };


    title.innerText =
        details[number].title;


    text.innerText =
        details[number].text;
}


/* ============================================================
   NAVIGATION
   ============================================================ */

function scrollToSection(id) {

    const element =
        document.getElementById(id);


    if (!element) {

        return;
    }


    element.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* ============================================================
   HTML ESCAPE
   ============================================================ */

function escapeHTML(value) {

    return String(value)

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );
}


/* ============================================================
   INITIAL LOAD
   ============================================================ */

loadHistory();

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


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction, confidence = \
            predict_news(text)


        print(
            "Prediction:",
            prediction
        )

        print(
            "Raw confidence:",
            confidence
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = float(
            confidence
        )


        # If predict.py returns 0-1,
        # convert to percentage.

        if (
            0 <= confidence <= 1
        ):

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


        # ====================================================
        # SUSPICIOUS LANGUAGE
        # ====================================================

        explanation =get_explanation(text)


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        save_prediction(

            text,

            str(prediction),

            confidence,

            explanation["message"]

        )


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return jsonify({

            "prediction":
                str(prediction),

            "confidence":
                confidence,

            "explanation":
                explanation

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
# INITIALIZE DATABASE
# ============================================================

init_database()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("FAKE NEWS DETECTION USING NLP")
    print("=" * 60)
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print()
    print("Model:")
    print("BERT + Knowledge Guided Analysis")
    print("=" * 60)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

