# Predictive Maintenance MLOps

Predicts whether a CNC milling machine will fail soon from 5 sensor readings, and serves the model as an API.

| File | Purpose |
|---|---|
| `data.py` | Generates the sensor dataset (modelled on AI4I 2020) |
| `train.py` | Trains candidate models, tracks them in MLflow, registers the best as `@production` |
| `app.py` | FastAPI service: `/predict`, `/health`, `/model-info`, docs at `/docs` |
| `tests/` | Data and API tests (pytest) |
| `Dockerfile` | Container for the API |
| `.github/workflows/ci.yml` | Train, test, build and smoke-test on every push |

```bash
pip install -r requirements.txt
python train.py && python -m pytest -q
uvicorn app:app --reload                          # http://127.0.0.1:8000/docs
docker build -t machine-failure-api . && docker run -p 8000:8000 machine-failure-api
```

`/predict` needs an `X-API-Key` header matching the `API_KEY` environment variable (without it every request gets 401), and is limited to `RATE_LIMIT` (default `3/second`) per client IP:

```bash
export API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo $API_KEY                                     # copy it for Swagger's Authorize button
gh secret set API_KEY --body "$API_KEY"           # same key for CI (or: repo Settings > Secrets and variables > Actions)
uvicorn app:app                                   # or: docker run -p 8000:8000 -e API_KEY=$API_KEY machine-failure-api
```

In Swagger (`/docs`), click Authorize, paste the key, then Try it out on `POST /predict`. Without the key it returns 401, and more than 3 calls in a second return 429.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"air_temp_k":300,"process_temp_k":310.5,"rotational_speed_rpm":1550,"torque_nm":62,"tool_wear_min":230}'
```
