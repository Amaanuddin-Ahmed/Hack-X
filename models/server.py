"""Local prediction service for the three Vitalis scikit-learn models.

Run with: uvicorn models.server:app --port 8000
"""
import base64
import binascii
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .xray_schemas import XrayAnalysis, XrayRequest
from .xray_service import ImageValidationError, analyze_chest_xray

ROOT = Path(__file__).resolve().parent
HEART = joblib.load(ROOT / "heart_random_forest.pkl")
OBESITY = joblib.load(ROOT / "obesity_random_forest.pkl")
DIABETES = joblib.load(ROOT / "diabetes_random_forest (1).pkl")

app = FastAPI(title="Vitalis model service", version="1.0.0")


class Profile(BaseModel):
    gender: Literal["Female", "Male"]
    age: float = Field(ge=18, le=77)
    height: float = Field(ge=120, le=220)
    weight: float = Field(ge=25, le=300)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    smoking_history: Literal["No Info", "current", "ever", "former", "never", "not current"]
    HbA1c_level: float = Field(ge=3.5, le=9)
    blood_glucose_level: int = Field(ge=80, le=300)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=94, le=200)
    chol: int = Field(ge=126, le=564)
    fbs: Literal[0, 1]
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=71, le=202)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=6.2)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)
    overweight_obese_family: Literal[1, 2]
    fast_food: Literal[1, 2]
    vegetables: int = Field(ge=1, le=3)
    main_meals: int = Field(ge=1, le=3)
    between_meals: int = Field(ge=1, le=4)
    liquid_intake: int = Field(ge=1, le=3)
    calorie_tracking: Literal[1, 2]
    physical_exercise: int = Field(ge=1, le=5)
    technology_time: int = Field(ge=1, le=3)
    transportation: int = Field(ge=1, le=5)


def positive_probability(model, frame: pd.DataFrame, positive_class: int = 1) -> float:
    probabilities = model.predict_proba(frame)[0]
    index = list(model.classes_).index(positive_class)
    return round(float(probabilities[index]) * 100, 1)


@app.get("/health")
def health():
    return {"status": "ok", "models": ["heart", "obesity", "diabetes"]}


@app.post("/predict")
def predict(profile: Profile):
    p = profile.model_dump()
    bmi = round(p["weight"] / (p["height"] / 100) ** 2, 1)

    # One friendly choice is bound to each model's training encoding.
    sex_heart = 1 if p["gender"] == "Male" else 0
    sex_obesity = 2 if p["gender"] == "Male" else 1
    smoking_obesity = 1 if p["smoking_history"] in {"current", "ever", "former"} else 2

    heart = pd.DataFrame([{
        "age": p["age"], "sex": sex_heart, "cp": p["cp"],
        "trestbps": p["trestbps"], "chol": p["chol"], "fbs": p["fbs"],
        "restecg": p["restecg"], "thalach": p["thalach"], "exang": p["exang"],
        "oldpeak": p["oldpeak"], "slope": p["slope"], "ca": p["ca"], "thal": p["thal"],
    }], columns=HEART.feature_names_in_)
    obesity = pd.DataFrame([{
        "Sex": sex_obesity, "Age": p["age"], "Height": p["height"],
        "Overweight_Obese_Family": p["overweight_obese_family"],
        "Consumption_of_Fast_Food": p["fast_food"],
        "Frequency_of_Consuming_Vegetables": p["vegetables"],
        "Number_of_Main_Meals_Daily": p["main_meals"],
        "Food_Intake_Between_Meals": p["between_meals"], "Smoking": smoking_obesity,
        "Liquid_Intake_Daily": p["liquid_intake"],
        "Calculation_of_Calorie_Intake": p["calorie_tracking"],
        "Physical_Excercise": p["physical_exercise"],
        "Schedule_Dedicated_to_Technology": p["technology_time"],
        "Type_of_Transportation_Used": p["transportation"],
    }], columns=OBESITY.feature_names_in_)
    diabetes = pd.DataFrame([{
        "gender": p["gender"], "age": p["age"], "hypertension": p["hypertension"],
        "heart_disease": p["heart_disease"], "smoking_history": p["smoking_history"],
        "bmi": bmi, "HbA1c_level": p["HbA1c_level"],
        "blood_glucose_level": p["blood_glucose_level"],
    }], columns=DIABETES.feature_names_in_)

    try:
        obesity_probabilities = OBESITY.predict_proba(obesity)[0]
        obesity_classes = list(OBESITY.classes_)
        # Dataset classes: 1 underweight, 2 normal, 3 overweight, 4 obesity.
        obesity_risk = sum(float(obesity_probabilities[obesity_classes.index(c)]) for c in (3, 4)) * 100
        scores = {
            "heart_disease": positive_probability(HEART, heart),
            "obesity": round(obesity_risk, 1),
            "diabetes": positive_probability(DIABETES, diabetes),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return {"scores": scores, "bmi": bmi}


@app.post("/xray", response_model=XrayAnalysis)
def xray(request: XrayRequest):
    try:
        payload = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image encoding.") from exc
    if not payload:
        raise HTTPException(status_code=400, detail="Empty X-ray image.")
    if len(payload) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="X-ray image must be 8 MB or smaller.")
    try:
        return analyze_chest_xray(payload, request.health_report)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Chest X-ray model is unavailable. Verify its dependencies and retry.") from exc
