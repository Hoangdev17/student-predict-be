import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load mô hình và scaler
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chỉ 16 features numeric mà model dùng
FEATURES = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
    'absences', 'G1', 'G2'
]

# Pydantic model chỉ chứa 16 feature này
class InputData(BaseModel):
    age: float
    Medu: float
    Fedu: float
    traveltime: float
    studytime: float
    failures: float
    famrel: float
    freetime: float
    goout: float
    Dalc: float
    Walc: float
    health: float
    absences: float
    G1: float
    G2: float


@app.post("/predict")
def predict(data: InputData):
    # Chuyển dataclass → list theo đúng thứ tự FEATURES
    values = [getattr(data, feature) for feature in FEATURES]

    arr = np.array([values])

    # Scale
    arr_scaled = scaler.transform(arr)

    # Predict
    pred = model.predict(arr_scaled)[0]

    return {
        "prediction": float(pred)
    }
