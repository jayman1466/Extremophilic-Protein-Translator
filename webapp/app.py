"""Extremophilic Protein Translator — job submission + results frontend.

Thin frontend: it validates input, records a job, and renders results. The heavy
generation pipeline runs elsewhere (Biotite SLURM now; serverless GPU later) and
writes a results.json + structure files into the job's storage dir. This app never
touches a GPU or a large database.
"""
from __future__ import annotations
import io
import json
import re

from flask import (Flask, render_template, request, redirect, url_for,
                   jsonify, send_file, abort, Response)

from pipeline_options import (SECTIONS, PHENOTYPES, default_selection,
                              validate_selection)
import aggressiveness as agg
import store
import backends

app = Flask(__name__)
store.init()

# Generation backend is chosen by env EPT_BACKEND (demo|slurm|broker); see
# backends.py. The public deployment runs `demo` (no credentials, no cluster
# path); the private in-network deployment runs `slurm`. Resolved lazily so an
# unavailable backend fails on first submit, not at import.
import os as _os
BACKEND_NAME = backends.selected_backend_name()

# Example enzyme for the "load example" button — B. subtilis lipase A (P37957),
# a small well-characterized secreted enzyme.
EXAMPLE_SEQ = ("MKFVKRRIIALVTILMLSVTSLFALQPSAKAAEHNPVVMVHGIGGASFNFAGIKSYLVSQGWSRDKLYAVDF"
               "WDKTGTNYNNGPVLSRFVQKVLDETGAKKVDIVAHSMGGANTLYYIKNLDGGNKVANVVTLGGANRLTTGKA"
               "LPGTDPNQKILYTSIYSSADMIVMNYLSRLDGARNVQIHGVGHIGLLYSSQVNSLIKEGLNGGGQNTN")

# IUPAC amino acids (20 standard + ambiguity codes B/Z/X, U selenocysteine, O pyrrolysine)
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBZXUO]+$", re.IGNORECASE)
MIN_LEN = 20
# ESM-2 positional limit is 1024 tokens = 1022 residues; generator + classifier run
# cleanly up to here. 1022-1500: accepted but ESM windows the sequence (cross-window
# contacts dropped) and ESMFold is memory-heavy — warn. >1500: ESMFold reliably OOMs
# and windowed generation is too lossy — hard reject.
ESM_WINDOW = 1022
MAX_LEN = 1500


def _clean_seq(raw: str) -> str:
    # strip FASTA header + whitespace/digits
    lines = [ln for ln in raw.splitlines() if not ln.startswith(">")]
    return re.sub(r"[\s0-9]", "", "".join(lines)).upper()


def validate_form(form):
    errs = []
    warnings = []
    title = (form.get("title") or "").strip()
    if not title:
        errs.append("Job title is required.")
    elif len(title) > 120:
        errs.append("Job title must be \u2264 120 characters.")

    seq = _clean_seq(form.get("sequence") or "")
    if not seq:
        errs.append("Amino-acid sequence is required.")
    elif not AA_RE.match(seq):
        bad = sorted(set(c for c in seq if not AA_RE.match(c)))
        errs.append(f"Sequence has invalid characters: {', '.join(bad)}")
    elif len(seq) < MIN_LEN:
        errs.append(f"Sequence too short ({len(seq)} aa; minimum {MIN_LEN}).")
    elif len(seq) > MAX_LEN:
        errs.append(f"Sequence too long ({len(seq)} aa; maximum {MAX_LEN}). "
                    f"ESMFold runs out of memory beyond this length.")
    elif len(seq) > ESM_WINDOW:
        warnings.append(
            f"Sequence is {len(seq)} aa, beyond ESM-2's {ESM_WINDOW}-residue window. "
            f"It will be processed in overlapping windows (long-range contacts that "
            f"span windows are dropped) and folding is memory-heavy. Results for "
            f"sequences \u2264 {ESM_WINDOW} aa are more reliable.")

    try:
        n_designs = int(form.get("n_designs") or 5)
        if not (1 <= n_designs <= 20):
            errs.append("Designs per phenotype must be between 1 and 20.")
    except ValueError:
        errs.append("Designs per phenotype must be a whole number.")
        n_designs = 5

    phenos = form.getlist("phenotypes")
    valid_ph = {p for p, _ in PHENOTYPES}
    if not phenos:
        errs.append("Select at least one phenotype to design for.")
    elif any(p not in valid_ph for p in phenos):
        errs.append("Unknown phenotype selected.")

    # pipeline selection
    selection = {}
    for key, _l, _h, multi, _opts in SECTIONS:
        selection[key] = request.form.getlist(key) if multi else request.form.get(key)
    errs.extend(validate_selection(selection))

    # advanced (expert) overrides — blank means "use the schedule"
    override = {}
    _bounds = {"adv_mask_rate": ("mask_rate", 0.02, 0.5),
               "adv_gamma": ("gamma", 0.5, 6.0),
               "adv_target_mut_frac": ("target_mut_frac", 0.01, 0.5)}
    for field, (key, lo, hi) in _bounds.items():
        raw = (form.get(field) or "").strip()
        if raw:
            try:
                v = float(raw)
                if not (lo <= v <= hi):
                    errs.append(f"{key} must be between {lo} and {hi}.")
                else:
                    override[key] = v
            except ValueError:
                errs.append(f"{key} must be a number.")
    selection["_advanced_override"] = override or None

    return errs, warnings, dict(title=title, sequence=seq, n_designs=n_designs,
                                phenotypes=phenos, selection=selection)


def _agg_ctx():
    return dict(agg_schedule={str(k): agg.schedule(k) for k in range(1, agg.N_LEVELS + 1)},
                agg_n_levels=agg.N_LEVELS)


@app.route("/")
def index():
    return render_template("index.html", sections=SECTIONS, phenotypes=PHENOTYPES,
                           defaults=default_selection(), form={}, errors=[],
                           example_seq=EXAMPLE_SEQ, **_agg_ctx())


@app.route("/submit", methods=["POST"])
def submit():
    errs, warnings, data = validate_form(request.form)
    if errs:
        return render_template("index.html", sections=SECTIONS, phenotypes=PHENOTYPES,
                               defaults=default_selection(), form=request.form,
                               errors=errs, example_seq=EXAMPLE_SEQ, **_agg_ctx()), 400
    jid = store.create_job(**data)
    if warnings:
        store.set_status(jid, "queued", message=" ".join(warnings))
    # Hand the job to the selected generation backend. `demo` synthesizes
    # results synchronously; `slurm`/`broker` set "running" and advance in poll.
    try:
        backends.get_backend().submit(store.get_job(jid))
    except Exception as e:  # noqa: BLE001
        store.set_status(jid, "error", message=f"submit failed: {e}")
    return redirect(url_for("job_view", jid=jid))


@app.route("/job/<jid>/cancel", methods=["POST"])
def cancel(jid):
    job = store.get_job(jid)
    if not job:
        abort(404)
    if job["status"] in ("queued", "running"):
        try:
            backends.get_backend().cancel(jid)  # best-effort remote scancel
        except Exception:  # noqa: BLE001 — cancel is best-effort
            pass
        store.set_status(jid, "cancelled", message="Cancelled by user.")
    return redirect(url_for("job_view", jid=jid))


@app.route("/jobs")
def jobs():
    return render_template("jobs.html", jobs=store.list_jobs())


@app.route("/job/<jid>")
def job_view(jid):
    job = store.get_job(jid)
    if not job:
        abort(404)
    results = store.load_results(jid) if job["status"] == "done" else None
    return render_template("job.html", job=job, results=results,
                           phenotype_labels=dict(PHENOTYPES))


@app.route("/api/job/<jid>/status")
def job_status(jid):
    job = store.get_job(jid)
    if not job:
        abort(404)
    # Let an async backend advance state (check remote, harvest) before reporting.
    if job["status"] in ("queued", "running"):
        try:
            backends.get_backend().poll(jid)
        except Exception:  # noqa: BLE001 — polling is best-effort
            pass
        job = store.get_job(jid)
    return jsonify(status=job["status"], message=job["message"])


@app.route("/job/<jid>/structure/<path:name>")
def structure_file(jid, name):
    p = store.job_dir(jid) / "structures" / name
    if not p.exists() or ".." in name:
        abort(404)
    return send_file(p, mimetype="chemical/x-pdb")


@app.route("/job/<jid>/download/<fmt>")
def download(jid, fmt):
    job = store.get_job(jid)
    results = store.load_results(jid) if job else None
    if not results:
        abort(404)
    if fmt == "tsv":
        buf = io.StringIO()
        # discover the union of nested metric keys across all designs, stable order
        metric_keys = []
        for designs in results["by_phenotype"].values():
            for d in designs:
                for mk in d.get("metrics", {}):
                    if mk not in metric_keys:
                        metric_keys.append(mk)
        base_cols = ["phenotype", "design_id", "classifier_score",
                     "active_site_rmsd", "n_mutations"]
        buf.write("\t".join(base_cols + metric_keys + ["sequence"]) + "\n")
        for ph, designs in results["by_phenotype"].items():
            for d in designs:
                row = [ph, d["design_id"], d.get("classifier_score", ""),
                       d.get("active_site_rmsd", ""), d.get("n_mutations", "")]
                row += [d.get("metrics", {}).get(mk, "") for mk in metric_keys]
                row.append(d.get("sequence", ""))
                buf.write("\t".join(str(x) for x in row) + "\n")
        return Response(buf.getvalue(), mimetype="text/tab-separated-values",
                        headers={"Content-Disposition": f"attachment; filename={jid}.tsv"})
    if fmt == "fasta":
        buf = io.StringIO()
        for ph, designs in results["by_phenotype"].items():
            for d in designs:
                buf.write(f">{jid}|{ph}|{d['design_id']} "
                          f"score={d.get('classifier_score','')} "
                          f"as_rmsd={d.get('active_site_rmsd','')}\n{d['sequence']}\n")
        return Response(buf.getvalue(), mimetype="text/x-fasta",
                        headers={"Content-Disposition": f"attachment; filename={jid}.fasta"})
    abort(404)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
