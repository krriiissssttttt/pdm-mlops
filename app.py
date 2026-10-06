"""FastAPI service that predicts machine failure from one sensor reading."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a). Replace "____" with that number, e.g. "0.3".
# THINK: two copies of one number can drift apart. How could the API read it from MLflow or metrics.json?
# DISCUSSION (open-ended): train.py already logs "threshold" as an MLflow parameter and writes metrics.json,
#   so the API could read it from the run of the @production model, or ship metrics.json in the image.
DEFAULT_THRESHOLD = "0.2"
if DEFAULT_THRESHOLD == "____":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_THRESHOLD to the threshold you chose in train.py")
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("machine-failure-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/machine-failure-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Machine Failure Prediction API", version="1.0.0",
              description="Predicts whether a milling machine will fail soon, from one sensor reading.")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible readings get a 422 error.
# One field is done for you as an example. Use df.describe() from Part 1:
#   air_temp_k            292.7 - 306.9    process_temp_k  301.8 - 317.9
#   rotational_speed_rpm  1170  - 2287     torque_nm       9.5   - 67.0     tool_wear_min  0 - 240
# Options for how wide to make the limits:
#   Option A: exactly the training min/max    -> safest for the model, but rejects some real readings
#   Option B: physical limits of the machine  -> e.g. torque 0-100 Nm, tool wear 0-300 min (needs domain input)
# THINK: what should happen to a reading that is possible but outside the training data? Reject? Warn?
# CHOSEN: Option A (the training min/max), rounded outward to a tidy number exactly like the given
#   air_temp_k example (training 292.7 - 306.9 became 290 - 310).
# DISCUSSION (open-ended): reject what is physically impossible (422). For a possible but never-seen value,
#   a real service would still answer but log a warning and flag the answer as low confidence, because the
#   model is guessing outside what it has seen.
class SensorReading(BaseModel):
    air_temp_k: float = Field(..., ge=290, le=310, description="Air temperature (K)", examples=[300.0])
    process_temp_k: float = Field(..., ge=300, le=320, description="Process temperature (K)", examples=[310.5])
    rotational_speed_rpm: float = Field(..., ge=1100, le=2300, description="Spindle speed", examples=[1550])
    torque_nm: float = Field(..., ge=5, le=70, description="Torque (Nm)", examples=[62.0])
    tool_wear_min: float = Field(..., ge=0, le=240, description="Tool wear (minutes)", examples=[230])


class Prediction(BaseModel):
    failure_probability: float
    failure_predicted: bool
    recommended_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    # HINT: the values are MODEL_SOURCE, THRESHOLD and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (an engineer debugging a strange prediction?)
    # DISCUSSION (open-ended): an engineer or a monitoring job asking "which model and threshold is this
    #   service really using?", e.g. after a deployment or when an operator doubts an alarm.
    return {"model_source": MODEL_SOURCE, "threshold": THRESHOLD, "features": FEATURES}


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    start = time.perf_counter()
    X = pd.DataFrame([reading.model_dump()])[FEATURES]
    probability = float(model.predict_proba(X)[0, 1])
    predicted = probability >= THRESHOLD

    # TODO 3b: turn the prediction into an action the machine operator can follow.
    #   Option A (2 levels): if predicted -> "Schedule maintenance this shift", else "No action needed"
    #   Option B (3 levels): also "Stop the machine and inspect now" when probability >= 0.7
    #   Option C (your own): e.g. include the probability in the message
    # THINK: who reads this on the shop floor? What would make them trust (or ignore) it?
    # CHOSEN: Option B (3 levels).
    # DISCUSSION (open-ended): an operator trusts a short, concrete instruction and ignores constant
    #   alarms. Reserving "Stop now" for very high probability keeps it credible.
    if probability >= 0.7:
        action = "Stop the machine and inspect now"
    elif predicted:
        action = "Schedule maintenance this shift"
    else:
        action = "No action needed"

    log.info("predict p=%.3f failure=%s latency_ms=%.1f", probability, predicted,
             (time.perf_counter() - start) * 1000)
    return Prediction(failure_probability=round(probability, 4),
                      failure_predicted=predicted, recommended_action=action)
