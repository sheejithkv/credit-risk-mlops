from fastapi.testclient import TestClient

from api.main import app, app_state


class DummyClassifier:
    def predict_proba(self, features):
        return [[0.8, 0.2]]


class DummyModel:
    named_steps = {"classifier": DummyClassifier()}

    def predict(self, features):
        return [0]

    def predict_proba(self, features):
        return [[0.8, 0.2]]

def valid_payload() -> dict:
    return {
        "age": 35,
        "duration_months": 24,
        "credit_amount": 5000,
        "installment_rate": 2,
        "existing_credits": 1,
        "people_liable": 1,
        "purpose": "car",
        "checking_account": "moderate",
        "savings": "low",
        "employment": "skilled",
        "personal_status": "single",
        "housing": "own",
        "job": "skilled",
        "property": "car",
        "other_installment_plans": "none",
        "foreign_worker": "yes",
        "telephone": "yes",
        "credit_history": "paid_back",
    }


def test_health() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready_returns_503_without_model() -> None:
    app_state["model"] = None
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503


def test_predict_success(monkeypatch) -> None:
    app_state["model"] = DummyModel()
    client = TestClient(app)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["risk_label"] == "good"
    assert body["probability_bad"] == 0.2
