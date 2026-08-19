// Sequence-track renderer: four orthogonal channels on one sequence so they
// never collide — conservation (background shade), active site (magma
// underline+dot below), interface (teal overline+dot above), mutation
// (orange/magma box). Exact values live in per-residue hover tooltips.
//
// Each track container carries a JSON payload in data-track:
//   { seq, wt, conservation:[floats|null], active_site:[1-based ints],
//     active_site_assigned:bool, mutations:[{pos,wt,mut}],
//     interfaces:[1-based ints],                  // union across §16b faces
//     interfaces_by_position:{ "pos": ["face_label", ...] } }

function consBg(c) {
  // pale (variable, c=0) -> deep magma (conserved, c=1). null -> no shade.
  if (c === null || c === undefined) return "transparent";
  const a = Math.max(0, Math.min(1, c));
  // interpolate white -> magma #5B1226 in RGB, capped alpha for readability
  const r = Math.round(255 + (91 - 255) * a);
  const g = Math.round(255 + (18 - 255) * a);
  const b = Math.round(255 + (38 - 255) * a);
  return `rgba(${r},${g},${b},${0.12 + 0.55 * a})`;
}

function renderTrack(el) {
  let d;
  try { d = JSON.parse(el.dataset.track); } catch (e) { return; }
  const seq = d.seq || "";
  const cons = d.conservation || [];
  const asSet = new Set(d.active_site || []);
  const ifaceSet = new Set(d.interfaces || []);
  const ifaceBy = d.interfaces_by_position || {};
  const mutMap = {};
  (d.mutations || []).forEach(m => { mutMap[m.pos] = m; });

  const frag = document.createDocumentFragment();

  // active-site warning
  if (!d.active_site_assigned) {
    const w = document.createElement("div");
    w.className = "as-warning";
    w.innerHTML = "&#9888; Active-site residues could not be assigned for this "
      + "enzyme (no database match). RMSD gate falls back to whole-domain; "
      + "interpret mutations near putative catalytic regions with caution.";
    frag.appendChild(w);
  }

  const track = document.createElement("div");
  track.className = "seq-track";
  for (let i = 0; i < seq.length; i++) {
    const pos = i + 1;
    const span = document.createElement("span");
    span.className = "res";
    span.textContent = seq[i];
    // conservation background
    const c = (i < cons.length) ? cons[i] : null;
    span.style.backgroundColor = consBg(c);
    if (c !== null && c !== undefined && c > 0.55) span.style.color = "#FBF8F2";
    // channels
    const isAS = asSet.has(pos);
    const isIF = ifaceSet.has(pos);
    const mut = mutMap[pos];
    if (isAS) span.classList.add("as");
    if (isIF) span.classList.add("iface");
    if (mut) span.classList.add("mut");
    // tooltip
    const parts = [`pos ${pos}`];
    if (mut) parts.push(`${mut.wt}\u2192${mut.mut} (mutated)`);
    else parts.push(seq[i]);
    if (c !== null && c !== undefined) parts.push(`conservation ${c.toFixed(2)}`);
    if (isAS) parts.push("active-site residue");
    if (isIF) {
      const faces = ifaceBy[String(pos)] || [];
      parts.push(faces.length ? `interface: ${faces.join(", ")}` : "interface residue");
    }
    span.setAttribute("data-bs-toggle", "tooltip");
    span.setAttribute("title", parts.join(" \u00b7 "));
    track.appendChild(span);
  }
  frag.appendChild(track);

  // legend
  const legend = document.createElement("div");
  legend.className = "track-legend";
  legend.innerHTML =
    '<span class="item"><span class="swatch" style="background:linear-gradient(90deg,#fff,#5B1226)"></span>conservation (light=variable, dark=conserved)</span>'
    + '<span class="item"><span class="swatch" style="border-bottom:3px solid #5B1226;height:0.5em"></span>active site</span>'
    + '<span class="item"><span class="swatch" style="border-top:3px solid #0F766E;height:0.5em"></span>interface (§16b)</span>'
    + '<span class="item"><span class="swatch" style="outline:2px solid #EE6C2B"></span>mutation</span>';
  frag.appendChild(legend);

  el.innerHTML = "";
  el.appendChild(frag);
  // enable tooltips for the residues just rendered
  if (window.bootstrap) {
    [...track.querySelectorAll('[data-bs-toggle="tooltip"]')]
      .forEach(x => new bootstrap.Tooltip(x));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".seq-track-data").forEach(renderTrack);
});
