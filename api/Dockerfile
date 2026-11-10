FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app.py model.tflite model_info.json ./

USER user
ENV PORT=10000
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/health')" || exit 1

CMD exec gunicorn -b 0.0.0.0:${PORT} -w 1 --timeout 120 app:app
