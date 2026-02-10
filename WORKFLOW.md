# Project Workflow & Contribution Guide

This document defines the **repository workflow, environment setup, and collaboration rules** for the AI-Assisted Predictive Maintenance project.

---

## 📁 Repository Structure

AI-Assisted-Predictive-Maintenance/
├── data/
│ ├── raw/ # Raw dataset (not tracked)
│ ├── processed/ # Cleaned / feature data
│ └── README.md
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_evaluation.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── models.py
│ ├── evaluation.py
│ └── utils.py
│
├── results/
│ ├── figures/
│ └── metrics.txt
│
├── pyproject.toml
├── poetry.lock
├── requirements.txt
├── README.md
└── WORKFLOW.md


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
