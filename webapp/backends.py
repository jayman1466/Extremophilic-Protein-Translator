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

import os
import subprocess
from abc import ABC, abstractmethod

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
    """Synthetic results, immediately. No credentials, no network — safe for a
    public deployment. Wraps the existing `make_demo_results` fixture."""

    name = "demo"

    def submit(self, job: dict) -> None:
        try:
            import make_demo_results
            override = (job.get("selection") or {}).get("_advanced_override")
            make_demo_results.main(
                job["id"], job["phenotypes"], job["n_designs"], override=override
            )
        except Exception as e:  # noqa: BLE001 — surface any fixture error on the job
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

    def submit(self, job: dict) -> None:
        # The generation pipeline (masked-LM + Gibbs + MPNN gate + ESMFold +
        # scoring) is not built yet. Until it is, be honest about state rather
        # than silently succeed. When wired, this method will: write the input
        # FASTA + selection to remote_root, sbatch submit_script, record the
        # remote job id (e.g. in the job message or a sidecar), set status
        # "running", and let poll() harvest results.json back into job_dir.
        store.set_status(
            job["id"], "error",
            message=("Generation pipeline not yet wired to SLURM. "
                     "SSH/SLURM plumbing is scaffolded; connect the generator here."),
        )

    def poll(self, jid: str) -> None:
        # When wired: squeue the recorded remote job id; on completion, scp/rsync
        # results.json + structures into store.job_dir(jid) and set "done".
        return None

    def cancel(self, jid: str) -> None:
        # When wired: scancel the recorded remote job id.
        return None


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
