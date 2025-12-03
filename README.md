# End-to-End Spam Detection Pipeline for SMS and Email

> **Production-Ready NLP Pipeline — Logistic Regression + Bag-of-Words + MLflow**

This project turns a notebook-style spam detection exercise into a **modular, production-ready NLP pipeline** for SMS and email messages.  

It follows the same architectural principles as the **Air Quality** project:

- clear separation of concerns,
- experiment tracking with **MLflow**,
- reproducible environments with **uv** and `pyproject.toml`,
- and automated testing via a teacher-provided `scripts/run_tests.py`.

The goal is to build a maintainable text classification system that can be used as a **portfolio-level ML engineering project**, not just a one-off notebook.

---

## 1. Learning Objectives

By completing this project, I demonstrate that I can:

- **NLP Engineering**
  - Preprocess raw text (tokenization, normalization, stop words, numbers)
  - Build bag-of-words feature representations with `CountVectorizer`
  - Train and evaluate a **logistic regression** spam classifier

- **Software Architecture**
  - Design modular components:
    - `DataProcessor`
    - `TextPreprocessor`
    - `ModelTrainer`
    - `Evaluator`
  - Keep each step testable and reusable

- **MLOps & Experimentation**
  - Track experiments with **MLflow** (parameters, metrics, artifacts)
  - Run multiple **dataset scenarios** (SMS only, Email only, Combined, Cross-domain)
  - Perform **statistical validation** with repeated runs and multiple seeds

- **Professional Practices**
  - Use `uv` for dependency and environment management
  - Organize code in a proper Python package (`src/…`)
  - Provide a clear, structured `README.md`

---

## 2. Data & Problem Statement

We tackle **binary spam detection** on two different text domains:

- **SMS dataset**
  - File: `data/sms_spam.csv`
  - Columns: `message`, `label`  
    - `label = 0` → ham (legitimate)
    - `label = 1` → spam

- **Email dataset**
  - File: `data/email_spam.csv`
  - Same schema: `message`, `label`

The main questions:

1. How well can a simple **bag-of-words + logistic regression** model detect spam?
2. How does preprocessing (tokenization, stop words, vocabulary size) affect performance?
3. How well does a model **transfer across domains** (train on SMS → test on email)?
4. Can we make these experiments **reproducible and inspectable** via MLflow?

---

## 3. Project Structure

The repository is organized as a modern Python project with `src/` layout:

```text
End-to-End-Spam-Detection-Pipeline-for-SMS-and-Email/
├─ data/
│  ├─ sms_spam.csv
│  └─ email_spam.csv
├─ mlruns/                     # MLflow tracking directory (auto-created)
├─ scripts/
│  ├─ run_experiments.py       # Grid of MLflow experiments (tokenization, features...)
│  ├─ run_significance_experiments.py
│  └─ run_tests.py             # Provided by teacher (not written by me)
├─ src/
│  ├─ pipeline/
│  │  ├─ __init__.py
│  │  ├─ data_processor.py     # Data loading + splitting across scenarios
│  │  ├─ text_preprocessor.py  # Text preprocessing + CountVectorizer
│  │  ├─ model_trainer.py      # LogisticRegression training
│  │  ├─ evaluator.py          # Metrics (accuracy, precision, recall, F1, confusion)
│  │  ├─ experiment_runner.py  # Utility wrapper for simple runs
│  │  └─ mlflow_runner.py      # Core MLflow integration (params, metrics, artifacts)
│  ├─ utils/
│  │  ├─ __init__.py
│  │  └─ config.py             # Paths & global constants (regex, nb_features, seeds…)
│  └─ spam_nlp/
│     └─ __init__.py           # Compatibility alias for teacher imports (→ pipeline.*)
├─ pyproject.toml
├─ uv.lock
└─ README.md

