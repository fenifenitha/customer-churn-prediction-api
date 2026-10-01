
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Load trained model and preprocessing files
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "content", "model")

model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_model.pkl")
)

target_encoder = joblib.load(
    os.path.join(MODEL_DIR, "city_target_encoder.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

feature_columns = joblib.load(
    os.path.join(MODEL_DIR, "feature_columns.pkl")
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([data])

    # Target encode City
    input_data["City"] = target_encoder.transform(
        input_data[["City"]]
    )

    # One-hot encode categorical columns
    categorical_cols = input_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_cols,
        drop_first=False
    )

    # Match training features
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert all values to float
    input_data = input_data.astype(float)

    # Apply same scaler used during training
    input_data = pd.DataFrame(
        scaler.transform(input_data),
        columns=feature_columns
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }
