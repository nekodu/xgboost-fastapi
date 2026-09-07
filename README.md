# XGBoost × FastAPI — model serving with a dynamic schema

Serves a pickled **XGBoost** model over a **FastAPI** REST endpoint. The interesting part: the request schema isn't hardcoded — at startup the API reads feature names and types straight from the model's booster and **generates the Pydantic input class dynamically**. Retrain the model with different features and the endpoint adapts without a single code change.

Built at DataSpecta to serve a CV-tuned XGBoost model (feature-importance-selected variables) as a REST API.

## How it works

- `mlapi.py` loads `xgboost_pickle.pkl`, reads `feature_names` / `feature_types` from the booster and builds an `InputFeatures` Pydantic model on the fly.
- `POST /predict` accepts a **JSON list** of feature objects (batch scoring) and returns the prediction list.
- Interactive Swagger docs are auto-generated at `/docs`.
- `rf_pickle.pkl` is an alternate RandomForest model kept in the repo (not wired into the API — swap the `pickle.load` line to serve it).

## Run

```bash
pip install -r requirements.txt
python mlapi.py
```

The server starts on `0.0.0.0:8080` and prints the expected feature dict on startup.

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '[{"feature_1": 1.0, "feature_2": 3.2}]'
```

(Replace keys with the feature names printed at startup.)

## Stack

Python · XGBoost · FastAPI · Pydantic · Uvicorn (Gunicorn-ready)
