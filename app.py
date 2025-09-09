import joblib, json
from fastapi import FastAPI
from pydantic import BaseModel

# Load artifacts
model = joblib.load("artifacts/model.joblib")
with open("artifacts/feature_order.json") as f:
    feature_order = json.load(f)

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Wine Quality API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: WineInput):
    data = [getattr(input, f) for f in feature_order]
    pred = model.predict([data])[0]
    prob = model.predict_proba([data])[0][1]
    return {
        "prediction": int(pred),
        "is_good_wine": bool(pred),
        "probability_good": float(prob)
    }
