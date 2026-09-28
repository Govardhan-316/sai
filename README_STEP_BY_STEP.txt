FAKE NEWS DETECTION USING NLP
Hybrid BERT + Knowledge-Guided Prompt + Progressive Training
===============================================================

IMPORTANT
---------
This project is a starting implementation for your mini project.
Do not claim a final accuracy until you actually train and evaluate it.

YOUR PROJECT FLOW
-----------------
WELFake Dataset
      |
      v
Text Cleaning
      |
      v
Knowledge-Guided Prompt Creation
      |
      v
BERT Tokenization
      |
      v
Progressive Training
  Stage 1 -> classifier head
  Stage 2 -> full BERT fine-tuning
      |
      v
Fake / Real
      |
      v
Accuracy, Precision, Recall, F1

STEP 1 - INSTALL PYTHON
-----------------------
Use Python 3.10 or 3.11 if possible.

STEP 2 - OPEN THIS FOLDER
-------------------------
Open this folder in VS Code.

STEP 3 - CREATE VIRTUAL ENVIRONMENT
-----------------------------------
Windows PowerShell:

python -m venv venv
.\venv\Scripts\activate

If PowerShell blocks activation, use Command Prompt:

venv\Scripts\activate.bat

STEP 4 - INSTALL LIBRARIES
--------------------------
pip install -r requirements.txt

STEP 5 - ADD DATASET
--------------------
Download/obtain your WELFake_Dataset.csv.

Copy it into this project folder and make sure its name is:

WELFake_Dataset.csv

STEP 6 - CHECK DATASET COLUMNS
------------------------------
The training code expects columns similar to:

title
text
label

If your column names are different, change the column detection section
in train.py.

IMPORTANT:
Check whether your dataset uses:
0 = Fake
1 = Real

This project assumes that mapping.

STEP 7 - TRAIN
--------------
Run:

python train.py

The first run downloads BERT. Internet is required for this first run.

Training may take a long time on CPU.
If you have a CUDA-compatible NVIDIA GPU, PyTorch can train much faster.

STEP 8 - CHECK RESULTS
----------------------
At the end you will see:

accuracy
precision
recall
f1

Save these values for your review.

STEP 9 - TEST ONE NEWS ARTICLE
------------------------------
Run:

python predict.py

Paste a news article.

You will get:

Prediction: Fake/Real
Confidence: xx.xx%

STEP 10 - RUN WEB APP
---------------------
Run:

python app.py

Open the local address shown by Flask, normally:

http://127.0.0.1:5000

HOW TO EXPLAIN THE NEW IDEA TO YOUR MENTOR
------------------------------------------
"We started from the traditional TF-IDF and machine-learning pipeline,
but our proposed improvement is a Transformer-based hybrid approach.
We use BERT for contextual representation, a task-specific
knowledge-guided prompt to inject credibility-related hints, and
progressive training in two stages. First, the classification head is
trained while the BERT encoder is frozen. Then the encoder is unfrozen
and fine-tuned with a smaller learning rate. We compare this proposed
model with our baseline models using accuracy, precision, recall and F1."

DO NOT SAY
----------
"External knowledge guarantees that the article is true."

The knowledge hints in this starter implementation are linguistic/domain
signals. If your guide specifically requires a real external knowledge
base or knowledge graph, that component should be added as a separate
extension.

DO NOT CLAIM
------------
Do not claim 95.3% for your proposed model unless your own experiment
produces 95.3%.

Do not add percentages just to make the presentation look better.

RECOMMENDED EXPERIMENTS
-----------------------
Experiment 1: Logistic Regression + TF-IDF
Experiment 2: Naive Bayes + TF-IDF
Experiment 3: SVM + TF-IDF
Experiment 4: Random Forest + TF-IDF
Experiment 5: BERT baseline
Experiment 6: BERT + knowledge-guided prompt
Experiment 7: BERT + knowledge-guided prompt + progressive training

This comparison makes the project much stronger because you can show
whether each proposed component actually improves performance.
