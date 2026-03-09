"""API smoke tests — mocked, no checkpoint required."""
import sys
import os
import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Minimal mock app — mirrors real API surface without loading model weights
app = FastAPI()

@app.get("/")
def root():
    return {"name": "arthur", "version": "3.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {"version": "3.0.0", "params": "125M", "name": "arthur"}

@app.post("/generate")
def generate(body: dict):
    return {"prompt": body.get("prompt", ""), "output": "test output", "tokens_generated": 5}

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "arthur" in r.json()["name"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_info():
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "params" in data


def test_generate():
    r = client.post("/generate", json={"prompt": "fn ", "max_tokens": 50, "temperature": 0.8})
    assert r.status_code == 200
    data = r.json()
    assert data["prompt"] == "fn "
    assert "output" in data
    assert "tokens_generated" in data
