"""Job store + result-file storage — swappable backends.

DB: SQLAlchemy over a connection string. Default is SQLite at $DB_PATH (point it
at a GCS-FUSE-mounted path on Cloud Run). Set $DATABASE_URL to a Cloud SQL
Postgres DSN to upgrade with zero code change.

Files: result artifacts (pdb/cif/tsv/fasta) live under $DATA_ROOT/jobs/<job_id>/.
On Cloud Run this is the same GCS FUSE mount; locally it's a workspace dir.
"""
from __future__ import annotations
import os
import json
import time
import uuid
from pathlib import Path

from sqlalchemy import (create_engine, Column, String, Integer, Float, Text,
                        DateTime, func)
from sqlalchemy.orm import declarative_base, sessionmaker

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./webapp_data"))
DB_URL = os.environ.get("DATABASE_URL") or f"sqlite:///{os.environ.get('DB_PATH', DATA_ROOT / 'jobs.db')}"

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    sequence = Column(Text, nullable=False)
    n_designs = Column(Integer, nullable=False, default=5)
    phenotypes = Column(Text, nullable=False)      # json list
    selection = Column(Text, nullable=False)       # json dict (pipeline options)
    status = Column(String, nullable=False, default="queued")  # queued|running|done|error
    message = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    results_path = Column(String, default="")      # relative path to results.json


_engine = None
_Session = None


def init():
    global _engine, _Session
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
    _engine = create_engine(DB_URL, connect_args=connect_args, future=True)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(bind=_engine, future=True)


def _session():
    if _Session is None:
        init()
    return _Session()


def create_job(title, sequence, n_designs, phenotypes, selection) -> str:
    jid = uuid.uuid4().hex[:12]
    with _session() as s:
        s.add(Job(id=jid, title=title, sequence=sequence.upper().replace("\n", "").strip(),
                  n_designs=n_designs, phenotypes=json.dumps(phenotypes),
                  selection=json.dumps(selection), status="queued"))
        s.commit()
    job_dir(jid).mkdir(parents=True, exist_ok=True)
    return jid


def get_job(jid):
    with _session() as s:
        j = s.get(Job, jid)
        if j is None:
            return None
        return dict(id=j.id, title=j.title, sequence=j.sequence, n_designs=j.n_designs,
                    phenotypes=json.loads(j.phenotypes), selection=json.loads(j.selection),
                    status=j.status, message=j.message or "",
                    created_at=str(j.created_at), results_path=j.results_path or "")


def list_jobs(limit=50):
    with _session() as s:
        rows = s.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        return [dict(id=j.id, title=j.title, status=j.status, created_at=str(j.created_at),
                     n_phenotypes=len(json.loads(j.phenotypes))) for j in rows]


def set_status(jid, status, message=""):
    with _session() as s:
        j = s.get(Job, jid)
        if j:
            j.status = status
            j.message = message
            s.commit()


def job_dir(jid) -> Path:
    return DATA_ROOT / "jobs" / jid


def load_results(jid):
    p = job_dir(jid) / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    return None
