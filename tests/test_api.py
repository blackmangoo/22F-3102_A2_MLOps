import io
import os
import sys
from pathlib import Path
from PIL import Image
import pytest

api_dir = Path(__file__).parent.parent / "api"
sys.path.insert(0, str(api_dir))

os.environ["API_KEY"] = "test_secret_key"

try:
    from app import app
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="API dependencies (ai-edge-litert / flask) not installed")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert "service" in data


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "threshold" in data


def test_predict_missing_auth(client):
    response = client.post("/predict")
    assert response.status_code == 401
    assert "invalid or missing API key" in response.get_json()["error"]


def test_predict_wrong_auth(client):
    response = client.post("/predict", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401


def test_predict_missing_file(client):
    response = client.post("/predict", headers={"X-API-Key": "test_secret_key"})
    assert response.status_code == 400
    assert "send the image in a form field" in response.get_json()["error"]


def test_predict_invalid_image(client):
    data = {"file": (io.BytesIO(b"not an image"), "test.txt")}
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test_secret_key"},
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_predict_valid_image(client):
    img = Image.new("RGB", (64, 64), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {"file": (buf, "sample.jpg")}
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test_secret_key"},
        data=data,
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    res = response.get_json()
    assert res["label"] in ["AI-generated", "Real"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert "model_version" in res
