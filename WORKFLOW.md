# Project Workflow & Contribution Guide

This document defines the **repository workflow, environment setup, and collaboration rules** for the AI-Assisted Predictive Maintenance project.

---

## 📁 Repository Structure

```text
AI-Assisted-Predictive-Maintenance/
│
├── data/
│   ├── raw/                 # Raw IMS bearing dataset (not tracked in Git)
│   ├── processed/           # Cleaned data and extracted features
│   └── README.md            # Dataset description and usage rules
│
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Signal processing & feature extraction
│   ├── 03_model_training.ipynb # ML model training
│   └── 04_evaluation.ipynb  # Model evaluation and results
│
├── src/
│   ├── preprocessing.py     # Data cleaning and signal processing functions
│   ├── models.py            # Machine learning models
│   ├── evaluation.py        # Metrics and evaluation logic
│   ├── utils.py             # Shared helper functions
│   └── __init__.py
│
├── results/
│   ├── figures/             # Plots and visualizations
│   └── metrics.txt          # Model performance metrics
│
├── pyproject.toml           # Poetry project configuration
├── poetry.lock              # Locked dependency versions
├── requirements.txt         # pip-compatible dependency list
├── README.md                # Project overview
├── WORKFLOW.md              # Workflow, setup, and contribution guide
├── .gitignore               # Ignored files and folders
└── LICENSE                  # Project license
```

---

## 🛠️ Environment Setup

### Recommended: Poetry (Standard for Team)

```bash
git clone <repository-url>
cd AI-Assisted-Predictive-Maintenance
poetry install
poetry shell
```
Verify:
```bash
python --version
```

---

### Alternative: pip + venv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🌿 Git Branching Strategy

- main → stable, final version (protected)

- dev → integration branch

- Individual branches:

	- min-dev

	- hein-dev

	- htin-dev

---

### Standard Flow 

```bash
git checkout dev
git pull origin dev
git checkout -b yourname-dev
```

####After completing work:

1. Push your branch

2. Open Pull Request → dev

3. Maintainer reviews and merges

🚫 Do NOT push directly to main.

---

## 👥 Roles & Responsibilities

**Maintainer**

- Reviews pull requests

- Resolves conflicts

- Merges into dev and main

- Ensures repo consistency

**Contributors**

- Work only on notebooks/modules

- Follow project structure

- Write clear commit messages

- Keep experiments reproducible

---

## 📓 Notebook Rules

- Each notebook has a single purpose

- Do not modify others’ notebooks without agreement

- Heavy logic should move to src/ modules

- Keep outputs clean before committing

## 📌 Development Stages

- Exploratory Data Analysis (EDA)

- Signal preprocessing & feature extraction

- Machine learning model training

- Evaluation & interpretation

- Final documentation & presentation

## ⚠️ Data Handling Rules

- Raw dataset goes into data/raw/

- Never commit raw data to Git

- Processed features go into data/processed/

## ✅ Reproducibility

- Dependencies are managed via Poetry

- poetry.lock guarantees consistent environments

- Use fixed random seeds where applicable

---
