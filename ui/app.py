import io
import os
import sqlite3
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, g, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:10000").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "history.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
ALLOWED = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def is_postgres():
    return DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


def get_db():
    if "db" not in g:
        if is_postgres():
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(DATABASE_URL)
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS history (
                            id SERIAL PRIMARY KEY,
                            ts VARCHAR(32),
                            filename VARCHAR(255),
                            label VARCHAR(64),
                            confidence REAL,
                            model_version VARCHAR(128)
                        );
                    """)
                conn.commit()
                g.db = conn
                g.db_type = "postgres"
                return g.db
            except Exception as e:
                app.logger.warning(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, filename TEXT,
            label TEXT, confidence REAL, model_version TEXT)""")
        g.db = conn
        g.db_type = "sqlite"
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def error(msg, code):
    return render_template("error.html", msg=msg, code=code), code


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_configured": bool(API_URL),
        "database": "postgres" if is_postgres() else "sqlite"
    }


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return error("Please choose an image file.", 400)
    name = secure_filename(f.filename)
    if not name or name.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return error("Only PNG, JPG and WEBP files are allowed.", 400)
    data = f.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        return error("This file looks like an image but cannot be decoded.", 400)

    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        r = requests.post(
            f"{API_URL}/predict",
            headers=headers,
            files={"file": (name, data)},
            timeout=120
        )
    except requests.RequestException:
        return error("The model API is unreachable (it may be waking up, retry in a minute).", 502)

    if r.status_code != 200:
        return error(f"Model API error: {r.text[:200]}", 502)

    res = r.json()
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db()
    if getattr(g, "db_type", "sqlite") == "postgres":
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO history(ts, filename, label, confidence, model_version) VALUES(%s, %s, %s, %s, %s)",
                (ts_now, name, res["label"], res["confidence"], res["model_version"])
            )
        conn.commit()
    else:
        conn.execute(
            "INSERT INTO history(ts, filename, label, confidence, model_version) VALUES(?, ?, ?, ?, ?)",
            (ts_now, name, res["label"], res["confidence"], res["model_version"])
        )
        conn.commit()

    return render_template("result.html", r=res, name=name)


@app.get("/history")
def history():
    conn = get_db()
    if getattr(g, "db_type", "sqlite") == "postgres":
        from psycopg2.extras import RealDictCursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, ts, filename, label, confidence, model_version FROM history ORDER BY id DESC")
            rows = cur.fetchall()
    else:
        rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows)


@app.errorhandler(413)
def too_large(_):
    return error("File is larger than 5 MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return error("Page not found.", 404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
