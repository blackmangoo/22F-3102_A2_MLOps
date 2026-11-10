import io
import os
import sys
from pathlib import Path
from PIL import Image
import pytest

ui_dir = Path(__file__).parent.parent / "ui"
sys.path.insert(0, str(ui_dir))

os.environ["DB_PATH"] = ":memory:"

try:
    from app import app, get_db
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="UI dependencies (flask) not installed")


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            get_db()
        yield client


def test_ui_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"VisionGuard" in response.data
    assert b"Check image" in response.data


def test_ui_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


def test_ui_history_empty(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert b"Prediction history" in response.data


def test_ui_predict_no_file(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert b"Please choose an image file" in response.data


def test_ui_predict_invalid_extension(client):
    data = {"file": (io.BytesIO(b"content"), "document.pdf")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Only PNG, JPG and WEBP files are allowed" in response.data


def test_ui_404(client):
    response = client.get("/nonexistent-page-xyz")
    assert response.status_code == 404
