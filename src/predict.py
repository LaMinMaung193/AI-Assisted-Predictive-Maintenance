
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
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    return prediction, probability


if __name__ == "__main__":

    sample = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [299.2],
        "Process temperature [K]": [308.5],
        "Rotational speed [rpm]": [1378],
        "Torque [Nm]": [50.4],
        "Tool wear [min]": [220]
    })

    pred, prob = predict_machine_failure(sample)

    if pred == 1:
        print("\n\nMachine Failure Likely\n")
    else:
        print("\n\nMachine Operating Normally\n")

    print("Prediction:", pred)
    print("Failure Probability:", round(prob*100,2), "%\n\n")





