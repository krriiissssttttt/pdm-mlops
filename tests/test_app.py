import pytest
from fastapi.testclient import TestClient

from app import app, limiter

client = TestClient(app, headers={"X-API-Key": "test-key"})
anonymous = TestClient(app)
HEALTHY = {"air_temp_k": 300, "process_temp_k": 310.5, "rotational_speed_rpm": 1550,
           "torque_nm": 38, "tool_wear_min": 20}
RISKY = {**HEALTHY, "torque_nm": 62, "tool_wear_min": 230}     # worn tool under high torque


# ---------- given tests ----------
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info_has_threshold():                            # checks TODO 3c
    body = client.get("/model-info").json()
    assert {"model_source", "threshold", "features"} <= set(body)


def test_predict_returns_valid_response():
    r = client.post("/predict", json=HEALTHY)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["failure_probability"] <= 1
    assert isinstance(body["failure_predicted"], bool)


def test_risky_machine_scores_higher_than_healthy():
    healthy = client.post("/predict", json=HEALTHY).json()
    risky = client.post("/predict", json=RISKY).json()
    assert risky["failure_probability"] > healthy["failure_probability"]
    assert risky["failure_predicted"] is True


def test_actions_are_filled_in_and_differ():                    # checks TODO 3b
    healthy = client.post("/predict", json=HEALTHY).json()["recommended_action"]
    risky = client.post("/predict", json=RISKY).json()["recommended_action"]
    assert "TODO" not in (healthy + risky)
    assert healthy != risky


# ---------- your tests ----------

def test_out_of_range_input_is_rejected():
    # TODO 4a: send a reading with an impossible value and assert the API answers 422.
    # HINT: r = client.post("/predict", json={**HEALTHY, "torque_nm": -5}); assert r.status_code == 422
    # THINK: test both edges? (a value just below your minimum AND just above your maximum)
    # DISCUSSION (open-ended): yes, a limit can be wrong at either end. Here the torque limits are 5 - 70.
    for bad_torque in (-5, 4.9, 70.1):          # impossible, just below the minimum, just above the maximum
        r = client.post("/predict", json={**HEALTHY, "torque_nm": bad_torque})
        assert r.status_code == 422


def test_missing_field_is_rejected():
    # TODO 4b: remove one field from HEALTHY, send it, and assert 422.
    # HINT: reading = dict(HEALTHY); reading.pop("tool_wear_min")
    for field in HEALTHY:                        # remove each field in turn
        reading = {k: v for k, v in HEALTHY.items() if k != field}
        assert client.post("/predict", json=reading).status_code == 422


def test_my_own_idea():
    # TODO 4c (optional, recommended): write a test of your own. Ideas:
    #   - more tool wear never LOWERS the failure probability (a "behaviour" test)
    #   - a text value such as "torque_nm": "high" is rejected with 422
    #   - /predict answers in under 200 ms
    #   - the API never predicts failure for a brand-new tool at normal torque
    # my own tests: (1) a text value is rejected, (2) a brand-new tool at normal torque raises no alarm
    assert client.post("/predict", json={**HEALTHY, "torque_nm": "high"}).status_code == 422
    new_tool = client.post("/predict", json={**HEALTHY, "tool_wear_min": 0}).json()
    assert new_tool["failure_predicted"] is False


# ---------- access control ----------
def test_predict_without_api_key_is_rejected():
    assert anonymous.post("/predict", json=HEALTHY).status_code == 401


def test_predict_with_wrong_api_key_is_rejected():
    r = anonymous.post("/predict", json=HEALTHY, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_health_needs_no_api_key():
    assert anonymous.get("/health").status_code == 200


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3/minute")
    limiter.reset()
    codes = [client.post("/predict", json=HEALTHY).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limiter.reset()

