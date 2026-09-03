from fastapi.testclient import TestClient

from api.main import STATE, app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "classifier" in body
    assert "timely_classifier" in body
    assert "index" in body


def test_predict_without_model_returns_503(monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setitem(STATE, "classifier", None)
        response = client.post(
            "/predict",
            json={"narrative": "This is a long enough complaint narrative for validation."},
        )
    assert response.status_code == 503


def test_predict_with_stub_model():
    class StubModel:
        classes_ = ["Credit card", "Mortgage"]

        def predict_proba(self, texts):
            return [[0.8, 0.2]]

    with TestClient(app) as client:
        STATE["classifier"] = StubModel()
        response = client.post(
            "/predict",
            json={"narrative": "Unexpected credit card late fee after autopay failed."},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["product"] == "Credit card"
    assert body["confidence"] == 0.8
    assert len(body["alternatives"]) == 2


def test_predict_timely_with_stub_model():
    class StubTimely:
        def predict_proba(self, texts):
            return [[0.2, 0.8]]

    with TestClient(app) as client:
        STATE["timely_classifier"] = StubTimely()
        response = client.post(
            "/predict/timely",
            json={"narrative": "Company answered my mortgage complaint within a week."},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["timely_response"] is True
    assert body["probability_timely"] == 0.8
