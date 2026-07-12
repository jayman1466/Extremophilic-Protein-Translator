# Design portal (frontend)

Thin Flask frontend for the extremophilic design pipeline. Validates an enzyme
sequence, records a job, and renders results (nested accordion + Mol* structure
overlay + metrics). **No GPU, no large databases** — the generation engine runs on
Biotite/serverless and writes a results bundle the frontend reads. See
`../docs/interface_design.md` for the full architecture.

## Run locally
```bash
pip install -r requirements.txt
export DATA_ROOT=./webapp_data          # SQLite + job files land here
python app.py                            # http://localhost:8080
```

## Try the results UI without the real pipeline
```bash
# submit a job in the browser, copy its job id, then:
python make_demo_results.py <job_id>     # writes synthetic results.json + structures
```

## Deploy to Cloud Run (project: extremolith)
```bash
gcloud run deploy ept-portal --source webapp --region us-central1 \
  --set-env-vars DATA_ROOT=/mnt/gcs \
  --add-volume name=gcs,type=cloud-storage,bucket=<BUCKET> \
  --add-volume-mount volume=gcs,mount-path=/mnt/gcs
```
`store.py` uses SQLite on the mounted bucket by default; set `DATABASE_URL` to a
Cloud SQL Postgres DSN to upgrade with no code change.

## Files
- `app.py` — routes, validation, downloads
- `store.py` — job DB + result-file storage (swappable backend)
- `pipeline_options.py` — data-driven catalog of selectable models/databases
- `make_demo_results.py` — synthetic results + the results-bundle schema contract
- `templates/`, `static/` — Bootstrap UI (theme palette in `static/css/theme.css`)
