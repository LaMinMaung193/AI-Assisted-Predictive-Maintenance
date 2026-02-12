# 📊 Dataset Documentation  
## AI4I 2020 Predictive Maintenance Dataset

---

## 1️⃣ Dataset Overview

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the  
UCI Machine Learning Repository.

The dataset simulates real industrial machine conditions and failure scenarios
for predictive maintenance applications.

---

## 2️⃣ Source Information

- 📌 Source: UCI Machine Learning Repository  
- 🔗 Dataset Name: AI4I 2020 Predictive Maintenance Dataset  
- 📄 File: ai4i2020.csv  
- 📊 Instances: 10,000  
- 📈 Features: 14 columns  
- ❌ Missing Values: No  
- 📜 License: Creative Commons Attribution 4.0 (CC BY 4.0)

---

## 3️⃣ Dataset Structure

### Feature Columns

| Column Name | Description | Unit |
|-------------|-------------|------|
| UID | Unique row identifier | - |
| Product ID | Product category (L/M/H) | - |
| Type | Product quality type | - |
| Air temperature | Ambient temperature | K |
| Process temperature | Process temperature | K |
| Rotational speed | Motor rotational speed | rpm |
| Torque | Applied torque | Nm |
| Tool wear | Tool wear time | minutes |

---

### Target Columns

| Column Name | Description |
|-------------|------------|
| Machine failure | 1 = Failure occurred |
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

---

## 4️⃣ Failure Definition

The `Machine failure` label is set to 1 if **any** of the failure modes occur:

- Tool Wear Failure (TWF)
- Heat Dissipation Failure (HDF)
- Power Failure (PWF)
- Overstrain Failure (OSF)
- Random Failure (RNF)

This allows:

- Binary classification (Failure vs No Failure)
- Multi-class classification (Failure type)
- Failure cause analysis
- Explainable AI analysis

---

## 5️⃣ Why This Dataset Was Selected

This dataset was selected because:

- ✔ Represents industrial machine conditions
- ✔ Suitable for predictive maintenance modeling
- ✔ Clean and well-structured
- ✔ Appropriate dataset size (10,000 samples)
- ✔ Multiple failure modes for advanced analysis
- ✔ Widely used in academic research

---

## 6️⃣ Data Storage Structure in This Repository

```text
data/
├── raw/
│ └── ai4i2020.csv
|
├── interim/
│ └── (process datasets will be stored here)
│
├── processed/
│ └── (cleaned datasets will be stored here)
│
└── README.md
```

- `raw/` → Original dataset (never modified)
- `interim/` → Process and modified datasets
- `processed/` → Cleaned and transformed datasets

---

## 7️⃣ Data Handling Rules

To maintain reproducibility:

1. Never modify files inside `raw/`
2. All preprocessing must be saved into `processed/`
3. Document preprocessing steps inside:
   - notebooks/02_preprocessing.ipynb
   - src/preprocessing.py

---

## 8️⃣ Planned Machine Learning Tasks

- Binary Classification (Failure / No Failure)
- Failure Type Prediction
- Feature Importance Analysis
- Model Comparison
- Explainable AI (SHAP or similar)

---

## 9️⃣ Citation

If used in academic reporting:

Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.

---

## 🔟 Next Step in Workflow

After placing the dataset in `data/raw/`:

1. Start Exploratory Data Analysis (EDA)
2. Analyze class imbalance
3. Visualize correlations
4. Define modeling strategy

---