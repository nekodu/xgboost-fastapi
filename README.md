# XGBoost model served with FastAPI, schema read from the model

A small 2022 prototype that serves the churn model from [churn-xgboost](https://github.com/nekodu/churn-xgboost) as a REST API.

The part I still like: the request schema is not written by hand. At startup the API reads the feature names and types stored inside the XGBoost booster and builds a Pydantic model from them. If the model is retrained with different features, the API validates requests against the new feature list without any code change, and the schema in the Swagger docs updates with it.

## Skills shown

- **Model serving:** a trained XGBoost model exposed as a REST API with FastAPI and Uvicorn, with batch scoring.
- **Schema-driven validation:** Pydantic models generated at runtime from the model's own metadata, so the contract between model and API can't drift apart.
- **Python metaprogramming:** building classes with `type()` and resolving types dynamically.
- **ML engineering judgement:** the limitations below are the ones that matter when a model moves from notebook to production.

## How it works

```python
feature_names = model.get_booster().feature_names
feature_types = model.get_booster().feature_types
InputFeatures = type("InputFeatures", (BaseModel,), {name: default_for(t) for name, t in ...})

@app.post("/predict")
async def predict(rows: List[InputFeatures]): ...
```

- `mlapi.py` loads `xgboost_pickle.pkl` and builds the `InputFeatures` class from the booster's metadata.
- `POST /predict` takes a JSON list of rows, so several customers can be scored in one request, and returns one prediction per row.
- FastAPI generates interactive docs at `/docs` from that dynamic class.
- `rf_pickle.pkl` is the RandomForest from the same notebook. Swap the `pickle.load` line to serve it instead.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python mlapi.py            # starts on http://localhost:8080, prints the expected fields
```

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '[{"<feature_1>": 0, "<feature_2>": 1.5}]'
```

Use the field names printed at startup, or open `http://localhost:8080/docs`.

## What I'd improve

- **Pickle runs code when loaded.** Only load model files you trust. Today I would save the model with XGBoost's JSON format (`save_model`) and load it with `load_model`.
- **Type mapping is fragile.** Feature types are turned into Python types with `pydoc.locate`, which works for `int` and `float` but not for every type XGBoost can store (for example `q` or `c`). An explicit mapping with a clear error would be safer.
- **Fields are optional with a default of 0.** A missing field is silently scored as 0 instead of being rejected. Real input should fail validation.
- **The model must match the preprocessing.** The API expects features already encoded and scaled as in the notebook. A production version would ship the preprocessing pipeline together with the model.
- **No tests, auth or versioning.** It was a prototype. A real deployment needs input tests, a model version in the response and authentication.
