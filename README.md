# VisionGuard: AI vs Real Image Classification & MLOps Pipeline

[![VisionGuard CI](https://github.com/blackmangoo/22F-3102_A2_MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/blackmangoo/22F-3102_A2_MLOps/actions/workflows/ci.yml)
[![DagsHub Project](https://img.shields.io/badge/DagsHub-22F--3102__A2__MLOps-blue.svg)](https://dagshub.com/blackmangoo/22F-3102_A2_MLOps)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview
**VisionGuard** is an end-to-end, production-grade MLOps system that detects whether an uploaded image is AI-generated or authentic (real). The platform incorporates disciplined data auditing, data version control (DVC), experiment tracking via MLflow on DagsHub, model registry versioning, model optimization via TensorFlow Lite, containerized microservices (Flask UI + Inference API + PostgreSQL), and cloud deployment.

- **Developer:** Ammar Akbar
- **Roll Number:** 22F-3102
- **GitHub Repository:** [https://github.com/blackmangoo/22F-3102_A2_MLOps](https://github.com/blackmangoo/22F-3102_A2_MLOps)
- **DagsHub Project:** [https://dagshub.com/blackmangoo/22F-3102_A2_MLOps](https://dagshub.com/blackmangoo/22F-3102_A2_MLOps)
- **Live Model API:** [https://two2f-3102-a2-mlops.onrender.com](https://two2f-3102-a2-mlops.onrender.com)

---

## 🏛️ System Architecture

```text
 ┌────────────────┐       HTTP Multipart (X-API-Key)       ┌────────────────────────┐
 │   User / Web   ├─────────────────────────────────────────►│  Inference API (Flask)  │
 │  Browser Client│                                         │   ai-edge-litert / CPU │
 └───────┬────────┘                                         └───────────┬────────────┘
         │                                                              │
         ▼                                                              ▼
┌──────────────────┐                                        ┌────────────────────────┐
│  Flask Web UI    │                                        │  model.tflite (8.5 MB) │
│  (Port 5000)     │                                        │  Optimized MobileNetV2 │
└────────┬─────────┘                                        └────────────────────────┘
         │
         │ (Database Trail)
         ▼
┌───────────────────────────┐
│ PostgreSQL / SQLite DB    │
│ history: id, ts, filename,│
│ label, conf, model_version│
└───────────────────────────┘
```

---

## 📂 Repository Layout

```text
22F-3102_A2_MLOps/
├── .dvc/                   # DVC configuration pointing to DagsHub remote
│   ├── config
│   └── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions CI (flake8, pytest, Docker builds)
├── api/                    # Inference microservice
│   ├── app.py              # Flask TFLite serving app with API key authentication
│   ├── Dockerfile          # Slim container with non-root user & health check
│   ├── dockerignore
│   ├── model.keras         # Full Keras fine-tuned champion model
│   ├── model.tflite        # Quantized / optimized lightweight serving model
│   ├── model_info.json     # Champion metadata, threshold, run id
│   └── requirements.txt    # Serving dependencies
├── ui/                     # Web application frontend & history trail
│   ├── app.py              # Flask UI connecting to API & logging to DB
│   ├── Dockerfile          # Production UI container
│   ├── requirements.txt
│   └── templates/
│       ├── base.html       # Clean responsive shell
│       ├── index.html      # Image upload form
│       ├── result.html     # Prediction verdict, confidence, honest caveat
│       ├── history.html    # Queryable search and prediction trail
│       └── error.html      # User-friendly error handlers
├── src/                    # MLOps pipeline scripts
│   ├── audit.py            # SHA-256 deduplication and image integrity audit
│   ├── make_splits.py      # Stratified 70/15/15 train/val/test split generator
│   ├── eda.py              # Exploratory Data Analysis & distribution plots
│   ├── data.py             # tf.data pipeline with train-only augmentations
│   ├── train.py            # Model training (Baseline CNN & MobileNetV2 + MLflow)
│   ├── aug_figure.py       # Batch augmentation visualizer
│   ├── error_analysis.py   # Test set misclassification diagnostic tool
│   ├── tuner_toy.py        # Keras Tuner baseline on Fashion-MNIST
│   ├── tuner_vg.py         # Keras Tuner hyperparameter search for VisionGuard
│   ├── register_model.py   # MLflow model registry client
│   ├── load_champion.py    # Registry champion model loader & inference test
│   ├── export_champion.py  # Model exporter
│   ├── to_tflite.py        # TFLite converter and numerical parity verifier
│   └── check_registry.py   # Registry inspection script
├── tests/                  # Automated test suite
│   ├── test_api.py         # Inference API unit & auth tests
│   ├── test_ui.py          # UI routes & validation tests
│   └── test_pipeline.py    # Data splits & audit validation tests
├── figures/                # EDA and error analysis artifacts
├── splits/                 # Train, validation, and test split indices
├── audit.csv               # Complete 975-image dataset audit metadata
├── data.dvc                # DVC dataset pointer file
├── docker-compose.yml      # Multi-container orchestration (DB + API + UI)
├── requirements.txt        # Full development environment requirements
├── .env.example            # Environment variable configuration template
└── README.md               # Project documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Clone the repository and install dependencies in an isolated virtual environment:
```bash
git clone https://github.com/blackmangoo/22F-3102_A2_MLOps.git
cd 22F-3102_A2_MLOps
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Data Audit & Versioning (DVC)
Audit the dataset for corrupted files and SHA-256 duplicate copies:
```bash
python src/audit.py
python src/make_splits.py
python src/eda.py
```
Track dataset versions with DVC and push to DagsHub remote storage:
```bash
dvc remote add origin https://dagshub.com/blackmangoo/22F-3102_A2_MLOps.dvc
dvc push
```

### 3. Model Training & MLflow Experiment Tracking
Train the baseline CNN and fine-tune MobileNetV2 with MLflow tracking:
```bash
# Baseline CNN
python src/train.py --model cnn --run_name cnn_base --epochs 15

# Fine-tuned MobileNetV2 (Stage 1 frozen backbone, Stage 2 unfrozen last block)
python src/train.py --model mobilenet --run_name mnet_finetune --epochs 15 --ft_epochs 10 --unfreeze 1
```

### 4. Model Registry & Champion Conversion
Register the best performing run as the Champion model in DagsHub MLflow Registry and export to TFLite:
```bash
python src/register_model.py --best mnet_finetune --second mnet_frozen
python src/to_tflite.py
```

### 5. Running the Automated Test Suite
Execute the full test suite with pytest:
```bash
pytest tests/ -v
```

---

## 🐳 Containerization with Docker Compose

Run the complete multi-service stack with a single command:
```bash
docker compose up --build
```
- **Web UI:** Accessible at `http://localhost:5000`
- **Inference API:** Accessible at `http://localhost:10000`
- **PostgreSQL Database:** Secured inside internal network `visionguard-net` (no outside host port published)
- **Health Checks:** Web UI waits for the database `pg_isready` check before accepting user traffic.

---

## 🌐 Production Cloud Deployment & API Contract

The inference API is deployed as a Dockerized web service on Render's cloud infrastructure:
- **Base URL:** `https://two2f-3102-a2-mlops.onrender.com`

### Health Check
```bash
curl -X GET https://two2f-3102-a2-mlops.onrender.com/health
```
**Response:**
```json
{
  "status": "ok",
  "model": "VisionGuard v2 (champion)",
  "threshold": 0.36
}
```

### Image Inference Request
```bash
curl -X POST https://two2f-3102-a2-mlops.onrender.com/predict \
  -H "X-API-Key: visionguard_secret_key" \
  -F "file=@sample_image.jpg"
```
**Response:**
```json
{
  "label": "AI-generated",
  "confidence": 0.9421,
  "p_ai": 0.9421,
  "threshold": 0.36,
  "model_version": "VisionGuard v2 (champion)"
}
```

---

## 🧪 CI/CD Pipeline
Continuous Integration is configured via **GitHub Actions** (`.github/workflows/ci.yml`):
1. **Linting:** Automated `flake8` syntax and code style verification.
2. **Testing:** Automated `pytest` execution across unit, integration, and data validation suites.
3. **Docker Build Validation:** Multi-stage image build checks for both the Inference API and Flask Web UI.
