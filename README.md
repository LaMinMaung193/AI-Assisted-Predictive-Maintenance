# AI-Assisted Predictive Maintenance System 

## Project Overview
This project presents an **AI-Assisted Predictive Maintenance System** designed to analyze machine operating data and **predict potential failures before they occur**.

The system applies machine learning techniques to sensor-based data collected from industrial equipment such as motors, pumps, and conveyor systems. By identifying patterns that indicate machine degradation, the model can support proactive maintenance decisions, reducing downtime and operational costs.

This project reflects the integration of Artificial Intelligence, Data Engineering, and Mechatronics systems, aligned with Industry 4.0 predictive maintenance practices.

Unlike traditional reactive or schedule-based maintenance, this project applies **data-driven intelligence** to monitor machine health and support decision-making in real-world industrial environments.

---

## Motivation
In industrial environments, unexpected machine failures can cause:

- Production downtime
- Increased maintenance cost
- Equipment damage
- Safety risks
- Reduced system efficiency

Traditional maintenance strategies are typically:

**Reactive Maintenance**

Repair after failure occurs.

**Preventive Maintenance**

Replace components based on schedule.


Both approaches are inefficient.

Predictive maintenance instead uses **data and AI models to anticipate failures**, allowing maintenance to be performed only when necessary.

---

## Problem Statement
How can we utilize sensor data from mechatronic systems to:
- Detect early signs of mechanical degradation
- Predict equipment failures accurately
- Assist maintenance planning using machine learning models

---

## Proposed Solution
The system collects and analyzes sensor data from machines, extracts meaningful features, and applies supervised machine learning models to classify the machine condition (e.g., healthy or faulty).
The system processes machine operating data through a complete machine learning pipeline.

### System Workflow
Sensors → Data Acquisition → Feature Extraction → AI Model → Maintenance Decision

### Project Architecture
```text
Raw Data
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Preprocessing & Feature Engineering
   │
   ▼
Class Imbalance Handling (SMOTEENN)
   │
   ▼
Model Training
(Logistic Regression, Decision Tree, Random Forest, SVM)
   │
   ▼
Model Evaluation
(Accuracy, Precision, Recall, F1, ROC-AUC)
   │
   ▼
Best Model Selection (Random Forest)
   │
   ▼
Model Serialization
   │
   ▼
Prediction Pipeline
(predict.py)
```

---

## Mechatronics Perspective
The project reflects a real-world mechatronics system by integrating:
- Mechanical components (motors, bearings)
- Sensors (vibration, temperature, current)
- Embedded systems (conceptual data acquisition)
- Artificial intelligence (machine learning models)

---

## Dataset
**Dataset used in this project:**

`AI4I 2020 Predictive Maintenance Dataset`

Location:

`data/raw/ai4i2020.csv`

The dataset contains simulated sensor readings from industrial machines (more about dataset in `data/README.md`)

**Other useable dataset:**

*(Public datasets such as bearing vibration or motor fault datasets can be used)*
- Represents vibration, temperature, or electrical signals from machinery
- Dataset is suitable for supervised learning

---

## Machine Learning Pipeline
The project implements a **complete ML workflow**.

**1. Exploratory Data Analysis**

Notebook:

``notebooks/01_exploratory_data_analysis.ipynb``

Tasks:

- dataset inspection

- correlation analysis

- feature distribution analysis

- imbalance analysis

---

**2. Preprocessing & Feature Engineering**

Notebook:

`notebooks/02_preprocessing_feature_engineering.ipynb`

Steps:

- categorical encoding

- feature scaling

- engineered features

- preprocessing pipeline creation

- dataset balancing using SMOTEENN

Output artifacts:
`
data/processed/
binary_dataset.pkl
multiclass_dataset.pkl
`

---

**3. Model Training**

Notebook:

`notebooks/03_model_training.ipynb`

Models implemented:

- Logistic Regression

- Decision Tree

- Random Forest

- Support Vector Machine (SVM)

---

**4. Model Evaluation**

Metrics used:

- Accuracy

- Precision

- Recall

- F1 Score

- ROC-AUC

---

## Deployment-Ready Artifacts

Saved models:
```text
artifacts/
├── preprocessing_pipeline.pkl
├── best_model.pkl
└── best_model_multiclass.pkl
```

These allow the system to perform predictions on new machine data.

---

## Prediction Pipeline

Prediction script:

`src/predict.py`

Example usage:
```bash
import pandas as pd
from predict import predict_machine_failure

sample = pd.DataFrame({
    "Type": ["L"],
    "Air temperature [K]": [298],
    "Process temperature [K]": [308],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [40],
    "Tool wear [min]": [120]
})

prediction = predict_machine_failure(sample)

print(prediction)

```
Output:

[0]  → No machine failure predicted

---

## Project Structure
```text
AI-Assisted-Predictive-Maintenance
│
├── artifacts/              # Saved models and pipelines
|     
├── data/
│   ├── raw/                # Original dataset
│   ├── interim/            # Cleaned intermediate data
│   └── processed/          # Final ML-ready datasets
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 03b_model_training_Multiclass.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   ├── predict.py
│   └── utils.py
│
├── results/
│   └── figures/
│
├── README.md
├── WORKFLOW.md
├── pyproject.toml
└── LICENSE
```


## AI Techniques Used
The following machine learning models are implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Learning Type
- **Supervised Learning**
- Classification-based fault detection

---

## Tools & Technologies

Programming & ML:

- Python

- Scikit-learn

- Pandas

- NumPy

Visualization:

- Matplotlib

- Seaborn

Development:

- Poetry (dependency management)

- Jupyter Notebook

- Git & GitHub

---

## Real-World Applications

This system can be applied to:

- industrial motors

- manufacturing production lines

- conveyor systems

- robotic systems

- smart factories

Predictive maintenance is widely used in:

- Industry 4.0

- smart manufacturing

- IoT-based monitoring systems

---

## Future Improvements
- Real-time sensor integration (ESP32 / PLC)

- Edge deployment for embedded systems

- Deep learning for time-series sensor data

- Anomaly detection models

- Real-time monitoring dashboard

- API deployment for industrial integration

---

## 👤 Author & Contributors
1. Kaung Hein San
2. Swe Mar Min Htin
3. La Min Maung

**Field:** **Electrical Engineering (Mechatronics)**

**Affiliation**: King Mongkut’s Institute of Technology Ladkrabang (KMITL), Thailand

---

## Conclusion
This project demonstrates how artificial intelligence can be effectively integrated into mechatronic systems to solve real industrial problems. By combining engineering knowledge with machine learning, the system provides a practical and scalable solution for predictive maintenance in modern industries.

---

## License
This project is developed for academic purposes under the Artificial Intelligence course.
