import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "aether" in response.json()["name"]

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert data["params"] == "3.5M"

def test_generate():
    response = client.post("/generate", json={
        "prompt": "fn ",
        "max_tokens": 50,
        "temperature": 0.8
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "fn "
    assert "output" in data
    assert "tokens_generated" in data

if __name__ == "__main__":
    test_root()
    test_health()
    test_info()
    test_generate()
    print("✓ All tests passed")
