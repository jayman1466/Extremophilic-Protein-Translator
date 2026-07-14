"""Generation backends — the boundary that keeps Biotite resources off the
public web.

The Flask app talks ONLY to the `Backend` interface; it never touches SSH keys,
sbatch, or cluster hosts directly. Which concrete backend is live is chosen by
one env var, `EPT_BACKEND`:

    demo    -> DemoBackend    synthetic results, no credentials, no network.
                              This is what the PUBLIC Cloud Run deployment runs.
    slurm   -> SlurmBackend   submits real jobs to Biotite over SSH+SLURM.
                              Only usable INSIDE the campus network (needs the
                              SSH key + host env). This is the PRIVATE demo
                              deployment (Tier 1): same codebase, run on your
                              laptop over VPN / a lab VM / a login node.
    broker  -> BrokerBackend  (future, Tier 3) writes an authorized job record
                              to a shared store; a poller running inside the
                              network picks it up. Cloud Run holds no secret and
                              has no inbound route. Stubbed for now.

Security invariant (Tier 1): the public app runs `demo` and holds no
credentials. `SlurmBackend` REFUSES TO CONSTRUCT unless the SSH key + host env
are present, so even if someone set `EPT_BACKEND=slurm` on the public instance
it would fail fast rather than expose a path to the cluster. The gate is at the
credential/network boundary, never a UI flag.

Back-compat: the older `EPT_DEMO` switch is still honored — `EPT_DEMO=0` with no
explicit `EPT_BACKEND` selects `slurm`.
"""
from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import store


class Backend(ABC):
    """Contract every generation backend implements.

    A backend owns a job from submit through terminal state. It is responsible
    for updating the job's status via `store.set_status(...)` and, on success,
    writing `results.json` (+ any structures) into `store.job_dir(jid)` in the
    results-bundle schema documented in `make_demo_results.py`.
    """

    name: str = "base"

    #: True if this backend can run in the CURRENT deployment (creds/host present).
    #: Constructing an unavailable backend raises; `available()` lets callers
    #: check without catching.
    @classmethod
    def available(cls) -> bool:
        return True

    @abstractmethod
    def submit(self, job: dict) -> None:
        """Kick off generation for `job` (the dict from `store.get_job`).

        Synchronous backends (demo) may produce results.json before returning
        and set status "done". Asynchronous backends (slurm/broker) should set
        status "running", return promptly, and advance state in `poll`.
        """

    def poll(self, jid: str) -> None:
        """Advance an async job's state (check remote, harvest results).

        Called by the status endpoint before reporting status. No-op for
        synchronous backends.
        """
        return None

    def cancel(self, jid: str) -> None:
        """Best-effort cancel of any remote work. Status is set to 'cancelled'
        by the route; this hook is for scancel/broker-flag cleanup."""
        return None


class DemoBackend(Backend):
    """Serves a REAL cached generation run as the demo output. No credentials, no
    network — safe for a public deployment. Copies the bundled laccase fixture
    (`fixtures/laccase_demo/`, the actual results from Biotite job 82374b61:
    Swiss-Prot annotation transfer + Otsu active-site freeze + Henikoff-weighted
    conservation) into the job dir, filtered to the phenotypes the user selected.

    Set EPT_DEMO_SYNTHETIC=1 to fall back to the old randomized `make_demo_results`
    fixture (useful for exercising the "active site not assigned" warning path)."""

    name = "demo"

    FIXTURE = Path(__file__).resolve().parent / "fixtures" / "laccase_demo"

    def submit(self, job: dict) -> None:
        if os.environ.get("EPT_DEMO_SYNTHETIC", "0") == "1":
            return self._synthetic(job)
        try:
            import shutil
            src = json.loads((self.FIXTURE / "results.json").read_text())
            jd = store.job_dir(job["id"])
            (jd / "structures").mkdir(parents=True, exist_ok=True)
            # keep only the phenotypes this job asked for; if none overlap, serve all
            want = set(job.get("phenotypes") or [])
            by = {ph: ds for ph, ds in src["by_phenotype"].items()
                  if not want or ph in want} or src["by_phenotype"]
            # copy the structure files actually referenced (WT + each kept design)
            needed = {src["wt_structure"]}
            for designs in by.values():
                needed.update(d["structure_file"] for d in designs)
            for name in needed:
                fp = self.FIXTURE / "structures" / name
                if fp.exists():
                    shutil.copy(fp, jd / "structures" / name)
            out = dict(src, by_phenotype=by)
            (jd / "results.json").write_text(json.dumps(out, indent=2))
            store.set_status(job["id"], "done")
        except Exception as e:  # noqa: BLE001 — surface any fixture error on the job
            store.set_status(job["id"], "error", message=f"demo fixture failed: {e}")

    def _synthetic(self, job: dict) -> None:
        try:
            import make_demo_results
            override = (job.get("selection") or {}).get("_advanced_override")
            make_demo_results.main(
                job["id"], job["phenotypes"], job["n_designs"], override=override
            )
        except Exception as e:  # noqa: BLE001
            store.set_status(job["id"], "error", message=f"demo generation failed: {e}")


class SlurmBackend(Backend):
    """Submit real generation jobs to Biotite over SSH+SLURM.

    USABLE ONLY inside the campus network, where the SSH key and host env are
    present. Reads its connection config from env so no secret is ever baked
    into the image:

        EPT_SLURM_HOST   e.g. jayminp@igi.biotite.berkeley.edu
        EPT_SLURM_KEY    path to the SSH private key
        EPT_SLURM_REMOTE_ROOT  remote scratch dir for job inputs/outputs
        EPT_SLURM_SUBMIT_SCRIPT  remote sbatch that runs the generation pipeline

    The generation pipeline itself is not wired yet, so `submit` currently
    records that clearly rather than pretending to run. The SSH plumbing
    (`_run_remote`) is in place so wiring the pipeline is a localized change.
    """

    name = "slurm"

    @classmethod
    def available(cls) -> bool:
        host = os.environ.get("EPT_SLURM_HOST", "").strip()
        key = os.environ.get("EPT_SLURM_KEY", "").strip()
        return bool(host and key and os.path.exists(key))

    def __init__(self):
        # Fail fast if the deployment has no path to the cluster. This is the
        # security gate: the public image cannot construct this backend.
        if not self.available():
            raise RuntimeError(
                "SlurmBackend unavailable: EPT_SLURM_HOST / EPT_SLURM_KEY not set "
                "or key missing. This backend only runs inside the campus network."
            )
        self.host = os.environ["EPT_SLURM_HOST"].strip()
        self.key = os.environ["EPT_SLURM_KEY"].strip()
        self.remote_root = os.environ.get(
            "EPT_SLURM_REMOTE_ROOT",
            "/groups/cress/projects/jaymin/eptrans_scratch/webapp_jobs",
        )
        self.submit_script = os.environ.get("EPT_SLURM_SUBMIT_SCRIPT", "")

    def _run_remote(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command on the cluster login node over SSH (batch-mode, keyed)."""
        ssh = [
            "ssh", "-i", self.key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            self.host, cmd,
        ]
        return subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)

    # Remote sbatch that chains the generation stages across conda envs.
    PIPELINE_SBATCH = (
        "/groups/cress/projects/jaymin/eptrans_scratch/repo/"
        "scripts/slurm/11_generate_pipeline.sbatch"
    )

    def _remote_job_dir(self, jid: str) -> str:
        return f"{self.remote_root}/{jid}"

    def submit(self, job: dict) -> None:
        """Stage the input FASTA to the cluster, sbatch the pipeline, record the
        remote SLURM job id in the job message, and set status 'running'. poll()
        then harvests results.json + structures back into the local job_dir."""
        try:
            jid = job["id"]
            rdir = self._remote_job_dir(jid)
            seq = "".join(job["sequence"].split())
            phenos = " ".join(job.get("phenotypes") or ["thermophile"])
            ndes = int(job.get("n_designs") or 3)

            # 1) make remote job dir + write input.fasta (base64 to survive quoting)
            import base64
            fasta_b64 = base64.b64encode(f">query\n{seq}\n".encode()).decode()
            mk = self._run_remote(
                f"mkdir -p {rdir} && echo {fasta_b64} | base64 -d > {rdir}/input.fasta && "
                f"wc -c {rdir}/input.fasta"
            )
            if mk.returncode != 0:
                store.set_status(jid, "error",
                                 message=f"remote staging failed: {mk.stderr[:400]}")
                return

            # 2) submit the pipeline (env vars carry per-job params)
            sub = self._run_remote(
                f"GEN_JOBDIR={rdir} GEN_PHENOS='{phenos}' GEN_NDESIGNS={ndes} "
                f"sbatch --parsable --export=ALL,GEN_JOBDIR={rdir},"
                f"GEN_PHENOS='{phenos}',GEN_NDESIGNS={ndes} {self.PIPELINE_SBATCH}"
            )
            slurm_id = (sub.stdout or "").strip().split(";")[0]
            if sub.returncode != 0 or not slurm_id.isdigit():
                store.set_status(jid, "error",
                                 message=f"sbatch failed: {(sub.stderr or sub.stdout)[:400]}")
                return

            # 3) record remote slurm id (encoded in message: 'slurm:<id>')
            store.set_status(jid, "running", message=f"slurm:{slurm_id}")
        except Exception as e:  # noqa: BLE001
            store.set_status(job["id"], "error", message=f"submit failed: {e}")

    @staticmethod
    def _slurm_id_from(job: dict) -> str:
        msg = job.get("message") or ""
        return msg.split("slurm:", 1)[1].strip() if "slurm:" in msg else ""

    def poll(self, jid: str) -> None:
        """squeue/sacct the recorded remote id; on completion, rsync results.json
        + structures/ into the local job_dir and set 'done' (or 'error')."""
        try:
            job = store.get_job(jid)
            if not job or job["status"] != "running":
                return
            slurm_id = self._slurm_id_from(job)
            if not slurm_id:
                return
            rdir = self._remote_job_dir(jid)

            # still queued/running?
            sq = self._run_remote(f"squeue -j {slurm_id} -h -o %T 2>/dev/null | head -1")
            state = (sq.stdout or "").strip()
            if state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING"):
                return  # keep waiting

            # terminal — did results.json land?
            chk = self._run_remote(f"test -f {rdir}/results.json && echo OK || echo MISSING")
            if "OK" not in (chk.stdout or ""):
                sac = self._run_remote(
                    f"sacct -j {slurm_id} -n -o State,ExitCode 2>/dev/null | head -1")
                store.set_status(jid, "error",
                                 message=f"pipeline ended without results ({sac.stdout.strip()})")
                return

            # harvest results.json + structures into local job_dir
            local = store.job_dir(jid)
            (local / "structures").mkdir(parents=True, exist_ok=True)
            for src, dst in ((f"{rdir}/results.json", local / "results.json"),):
                r = subprocess.run(
                    ["scp", "-i", self.key, "-o", "BatchMode=yes",
                     "-o", "StrictHostKeyChecking=accept-new",
                     f"{self.host}:{src}", str(dst)],
                    capture_output=True, text=True, timeout=120)
                if r.returncode != 0:
                    store.set_status(jid, "error",
                                     message=f"results scp failed: {r.stderr[:300]}")
                    return
            # structures dir (recursive)
            subprocess.run(
                ["scp", "-r", "-i", self.key, "-o", "BatchMode=yes",
                 "-o", "StrictHostKeyChecking=accept-new",
                 f"{self.host}:{rdir}/structures", str(local)],
                capture_output=True, text=True, timeout=300)
            store.set_status(jid, "done", message="")
        except Exception as e:  # noqa: BLE001
            store.set_status(jid, "error", message=f"poll failed: {e}")

    def cancel(self, jid: str) -> None:
        try:
            job = store.get_job(jid)
            slurm_id = self._slurm_id_from(job) if job else ""
            if slurm_id:
                self._run_remote(f"scancel {slurm_id}")
        except Exception:  # noqa: BLE001
            pass


class BrokerBackend(Backend):
    """Future (Tier 3): decouple entirely — write an authorized job record to a
    shared store; a poller running inside the network runs it and writes results
    back. Cloud Run then holds no secret and has no inbound route to Biotite.
    Stubbed until Tier 1 is outgrown."""

    name = "broker"

    def submit(self, job: dict) -> None:
        store.set_status(
            job["id"], "error",
            message="Broker backend not implemented (Tier 3 — future).",
        )


_REGISTRY = {b.name: b for b in (DemoBackend, SlurmBackend, BrokerBackend)}


def selected_backend_name() -> str:
    """Resolve which backend the deployment wants, honoring the legacy switch."""
    name = os.environ.get("EPT_BACKEND", "").strip().lower()
    if name:
        return name
    # back-compat: EPT_DEMO=0 historically meant "use the real backend"
    return "demo" if os.environ.get("EPT_DEMO", "1") != "0" else "slurm"


_INSTANCE: Backend | None = None


def get_backend() -> Backend:
    """Return the process-wide backend singleton for the selected mode.

    Raises if the selected backend is unknown or unavailable in this deployment
    (e.g. `slurm` selected without credentials) — a loud failure at startup is
    the point, so a misconfigured public instance never limps along with a live
    cluster path.
    """
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    name = selected_backend_name()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise RuntimeError(
            f"Unknown EPT_BACKEND={name!r}; expected one of {sorted(_REGISTRY)}."
        )
    _INSTANCE = cls()  # may raise for slurm without creds — intended
    return _INSTANCE
