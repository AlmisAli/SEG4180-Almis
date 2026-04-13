from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_invalid_input():
    client = app.test_client()
    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()