# Thin frontend only — no GPU, no large databases. The generation pipeline runs
# elsewhere and writes results into the GCS-mounted job dir.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PORT=8080
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DATA_ROOT should point at the GCS FUSE mount on Cloud Run (e.g. /mnt/gcs).
# gunicorn: 2 workers is plenty for a poll-based frontend.
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 app:app
