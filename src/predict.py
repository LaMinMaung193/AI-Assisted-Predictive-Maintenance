import joblib
import pandas as pd
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Artifact paths
preprocessor_path = os.path.join(BASE_DIR, "artifacts", "preprocessing_pipeline.pkl")
model_path = os.path.join(BASE_DIR, "artifacts", "best_model.pkl")

# Load artifacts
preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)


def predict_machine_failure(input_data: pd.DataFrame):

    # Apply preprocessing
    X_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(X_processed)

    return prediction


sample = pd.DataFrame({
    "Type": ["L"],
    "Air temperature [K]": [298],
    "Process temperature [K]": [308],
    "Rotational speed [rpm]": [1455],
    "Torque [Nm]": [40],
    "Tool wear [min]": [20]
})

result = predict_machine_failure(sample)

print("Prediction:", result)