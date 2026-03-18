# 🏥 Clinical NLP Classifier with LLM Evaluation

A two-part system that classifies clinical notes into medical specialties using 
a traditional ML pipeline, then uses GPT-4o-mini to evaluate predictions, 
diagnose failure modes, and identify boundary cases that standard metrics miss.

🔗 **[Live Demo](https://clinicalnlp.streamlit.app/)**

---

## Overview

Standard ML classifiers report a single accuracy number — but that number hides 
a lot. A Surgery note misclassified as Orthopedic is a very different kind of 
error than a Surgery note misclassified as Radiology. This project adds an LLM 
evaluation layer that reads each misclassification like a clinician would, 
explains why it failed, and identifies whether the error was a genuine mistake 
or a defensible boundary case.

**Key finding:** Raw classifier accuracy was 57.4%. LLM evaluation revealed that 
11 of 19 misclassifications were boundary cases between overlapping specialties 
— Surgery vs Orthopedic, Cardiovascular vs Radiology — where even human 
annotators might disagree. Adjusted accuracy accounting for label ambiguity: 
**73.3%.**

---

## Features

- 🔍 **Real-time classification** — paste any clinical note and get an instant 
  specialty prediction with confidence scores across all 5 classes
- 🤖 **LLM evaluation** — GPT-4o-mini evaluates the prediction, explains its 
  reasoning, identifies key clinical terms, and suggests a label if it disagrees
- 📊 **Evaluation dashboard** — aggregate metrics across a 30-sample test set 
  including classifier accuracy, LLM agreement, adjusted accuracy, and 
  boundary case analysis
- ⚖️ **Boundary case detection** — automatically identifies misclassifications 
  that are clinically defensible vs genuine errors

---

## Dataset

**MTSamples** — 4,999 real medical transcription notes across 40 specialties.
Filtered to 5 clinically distinct categories for this project:

| Specialty | Training Samples |
|---|---|
| Surgery | 882 |
| Cardiovascular / Pulmonary | 297 |
| Orthopedic | 284 |
| Radiology | 218 |
| Neurology | 178 |

Class imbalance handled via `class_weight="balanced"` in the classifier.

---

## Architecture
```
Clinical Note (raw text)
        │
        ▼
TF-IDF Vectorizer          ← 10,000 features, unigrams + bigrams
        │
        ▼
Logistic Regression        ← class_weight="balanced" for imbalance
        │
        ▼
Predicted Specialty        ← with confidence scores across all classes
        │
        ▼
GPT-4o-mini Evaluator      ← reads note like a clinician
        │
        ├── is_correct (true/false)
        ├── confidence score
        ├── reasoning (2-3 sentences)
        ├── key_clinical_terms
        └── suggested_label
```

---

## Results

| Metric | Value |
|---|---|
| Raw classifier accuracy | 57.4% |
| LLM agreement with true labels | 36.7% |
| Boundary/ambiguous misclassifications | 11/19 (58%) |
| Adjusted accuracy (excl. boundary cases) | 73.3% |
| Average LLM confidence | 0.89 |

**Key insight:** The majority of misclassifications were boundary cases between 
overlapping specialties — orthopedic surgery notes labeled as general surgery, 
cardiovascular procedures performed by radiologists, neurological symptoms 
evaluated via imaging. A confusion matrix alone cannot surface this distinction. 
LLM evaluation can.

---

## Tech Stack

| Component | Tool |
|---|---|
| Dataset | MTSamples (Kaggle) |
| Vectorization | scikit-learn TfidfVectorizer |
| Classifier | scikit-learn LogisticRegression |
| LLM Evaluator | OpenAI GPT-4o-mini |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/hanlinm/clinical_nlp.git
cd clinical_nlp
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv clinical_env
source clinical_env/bin/activate  # Windows: clinical_env\Scripts\activate
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
```

**4. Download the dataset**

Download `mtsamples.csv` from 
[Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) 
and place it in the `data/` folder.

**5. Train the classifier**
```bash
python classifier.py
```

**6. Run LLM evaluation**
```bash
python evaluator.py
```

**7. Launch the app**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## Project Structure
```
clinical_nlp/
├── classifier.py        # TF-IDF + Logistic Regression training pipeline
├── evaluator.py         # LLM evaluation framework using GPT-4o-mini
├── app.py               # Streamlit two-tab interface
├── requirements.txt
├── data/
│   └── mtsamples.csv    # source dataset (download from Kaggle)
└── .env                 # not committed — see above
```

---

## Key Design Decisions

**TF-IDF over embeddings for the classifier** — clinical notes have highly 
distinctive domain vocabulary. TF-IDF with bigrams captures multi-word clinical 
terms ("myocardial infarction", "anterior cruciate") effectively without the 
cost and complexity of embedding-based approaches. For a production system, 
a fine-tuned BioBERT would be the next step.

**class_weight="balanced"** — Surgery has 5x more samples than Neurology. 
Without balancing, the classifier learns to over-predict Surgery. Balanced 
weighting treats each class as equally important during training regardless 
of sample count.

**temperature=0 on the LLM evaluator** — evaluation requires consistent, 
deterministic reasoning. Temperature 0 eliminates response variation so 
the same note always produces the same evaluation — essential for a reliable 
evaluation framework.

**Sampling more misclassifications than correct predictions** — the evaluation 
set intentionally oversamples errors (20 misclassifications vs 10 correct). 
Correct predictions are less interesting to analyze; understanding failure 
modes is where the value is.

**Boundary case detection via keyword matching** — the LLM reasoning text 
consistently uses hedging language ("reasonable", "while", "although") when 
acknowledging ambiguous cases. Simple keyword matching on this text is a 
lightweight but effective signal for identifying boundary cases.

---

## Future Improvements

- [ ] Replace TF-IDF with BioBERT embeddings for richer clinical representations
- [ ] Expand to all 40 MTSamples specialties
- [ ] Add confidence calibration analysis
- [ ] Implement active learning loop — use LLM evaluation to identify 
  high-value samples for retraining
- [ ] Add LangSmith tracing for evaluation quality monitoring
- [ ] Export evaluation reports as PDF

---

## About

Built as part of a portfolio project to demonstrate applied ML engineering 
combining traditional NLP pipelines with LLM evaluation frameworks. Domain 
focus reflects 5+ years of professional experience in computational biology, 
healthtech, and drug discovery.