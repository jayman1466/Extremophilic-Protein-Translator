#!/usr/bin/env python
"""Standalone sequence-track exporter.

Renders the same four channels as the Flask webapp (conservation shade,
active-site underline+dot, interface (§16b) overline+dot, mutation outline)
but as a static SVG/PNG figure — no server required.

Input formats supported:
  --results results.json         (webapp bundle; renders every design of every phenotype)
  --candidates candidates.json   (11_generate.py output; renders every design in one file)
  --candidates cand.json --folded folded.json --mpnn mpnn.json   (webapp assembly on the fly)

Outputs one SVG per design at <out_dir>/<pheno>_<design_id>.svg (plus a .png sibling
if --png is passed). Colors match webapp/static/css/theme.css and track.js.

Example:
  # single candidates.json (per-phenotype generation output)
  python scripts/render_track_svg.py \
      --candidates gen/is621_20260818_213941_thermophile/candidates.json \
      --out-dir figures/is621_thermophile_tracks --png

  # assembled results.json (webapp bundle)
  python scripts/render_track_svg.py \
      --results gen/is621_20260818_213941_thermophile/results.json \
      --out-dir figures/is621_thermophile_tracks
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

# ---- palette (matches webapp/static/css/theme.css) --------------------------
COL_CONS_DARK = "#5B1226"   # magma (conservation shade endpoint, and active site)
COL_ACTIVE     = "#5B1226"  # active-site underline+dot
COL_INTERFACE  = "#0F766E"  # teal overline+dot (§16b)
COL_MUTATION   = "#EE6C2B"  # orange outline
COL_MUT_AS     = COL_ACTIVE
COL_MUT_IF     = COL_INTERFACE
BG_LIGHT       = "#FFFFFF"
TEXT_DARK      = "#1a1a1a"
TEXT_LIGHT     = "#FBF8F2"

# ---- payload building -------------------------------------------------------

def _iface_summary(interfaces_faces: dict) -> tuple[list[int], dict[str, list[str]]]:
    """Collapse the candidates.json `interfaces` face dict to the flat form the
    renderer needs: sorted 1-based union positions + per-position face labels."""
    pos_set: set[int] = set()
    by_pos: dict[int, list[str]] = {}
    for label, face in (interfaces_faces or {}).items():
        for p in face.get("positions", []):
            pos_set.add(int(p))
            by_pos.setdefault(int(p), []).append(label)
    return sorted(pos_set), {str(k): v for k, v in by_pos.items()}


def _tracks_from_candidates(cand: dict) -> list[tuple[str, str, dict]]:
    """Yield (phenotype, design_id, track_payload) triples from a candidates.json."""
    ph = cand.get("phenotype", "phenotype")
    wt = cand["wt_sequence"]
    cons = cand.get("conservation", [])
    active = cand.get("active_site", [])
    assigned = cand.get("active_site_assigned", False)
    iface_positions, iface_by = _iface_summary(cand.get("interfaces", {}))
    out = []
    for d in cand["designs"]:
        track = dict(
            seq=d["sequence"], wt=wt,
            conservation=cons, active_site=active,
            active_site_assigned=assigned,
            mutations=d.get("mutations", []),
            interfaces=iface_positions,
            interfaces_by_position=iface_by,
        )
        out.append((ph, d["design_id"], track))
    return out


def _tracks_from_results(results: dict) -> list[tuple[str, str, dict]]:
    """Yield (phenotype, design_id, track_payload) from an assembled results.json."""
    out = []
    for ph, rows in results.get("by_phenotype", {}).items():
        for r in rows:
            t = r.get("track")
            if t:
                out.append((ph, r["design_id"], t))
    return out


# ---- rendering --------------------------------------------------------------

def _cons_bg(c: float | None) -> tuple[float, float, float, float]:
    """white -> #5B1226 magma, alpha-capped for readability. Matches track.js consBg()."""
    if c is None:
        return (0, 0, 0, 0)
    a = max(0.0, min(1.0, float(c)))
    r = 255 + (91 - 255) * a
    g = 255 + (18 - 255) * a
    b = 255 + (38 - 255) * a
    return (r / 255, g / 255, b / 255, 0.12 + 0.55 * a)


def render_track(track: dict, out_path: Path, *, wrap: int = 60, cell_w: float = 0.16,
                 cell_h: float = 0.36, dpi: int = 200, also_png: bool = False,
                 title: str | None = None) -> None:
    """Render one track payload to an SVG (and optionally a sibling PNG).

    Layout: monospace grid, one column per residue, wrapping every `wrap` residues.
    Each cell paints (bottom to top): conservation shade -> letter -> active-site
    underline+dot (below cell) -> interface overline+dot (above cell) -> mutation
    outline. Row header shows the 1-based start position of that row.
    """
    seq = track["seq"]
    cons = track.get("conservation") or []
    active = set(track.get("active_site") or [])
    iface = set(track.get("interfaces") or [])
    mut_map = {int(m["pos"]): m for m in (track.get("mutations") or [])}
    iface_by = track.get("interfaces_by_position") or {}
    assigned = track.get("active_site_assigned", True)

    L = len(seq)
    n_rows = math.ceil(L / wrap)
    # figure geometry
    left_gutter = 0.55   # inches for the position label
    right_pad   = 0.15
    row_h_in    = cell_h + 0.30  # cell + underline/overline breathing room
    top_pad     = 0.55 if (title or not assigned) else 0.20
    bottom_pad  = 0.55   # legend
    fig_w = left_gutter + wrap * cell_w + right_pad
    fig_h = top_pad + n_rows * row_h_in + bottom_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_axis_off()

    # title / active-site warning
    if title:
        ax.text(left_gutter, fig_h - 0.20, title, fontsize=9, weight="bold",
                color=TEXT_DARK, va="top", ha="left")
    if not assigned:
        ax.text(fig_w - right_pad, fig_h - 0.20,
                "\u26a0 active-site not assigned (RMSD fallback: whole-domain)",
                fontsize=7, color="#7a5300", va="top", ha="right")

    # draw each residue
    for i, aa in enumerate(seq):
        pos = i + 1
        row = i // wrap
        col = i % wrap
        # cell bottom-left in inches, flipping so row 0 is at the top:
        x0 = left_gutter + col * cell_w
        y0 = fig_h - top_pad - (row + 1) * row_h_in + 0.15  # +0.15 leaves room below for AS dot

        # conservation shade
        c = cons[i] if i < len(cons) else None
        bg = _cons_bg(c)
        if bg[3] > 0:
            ax.add_patch(Rectangle((x0, y0), cell_w, cell_h, facecolor=bg,
                                   edgecolor="none", linewidth=0, zorder=1))
        # residue letter
        text_col = TEXT_LIGHT if (c is not None and c > 0.55) else TEXT_DARK
        weight = "bold" if pos in mut_map else "normal"
        # mutated letter uses the mutant identity (as in the JS renderer)
        letter = mut_map[pos]["mut"] if pos in mut_map else aa
        ax.text(x0 + cell_w / 2, y0 + cell_h / 2, letter,
                fontsize=7.5, family="monospace", color=text_col,
                weight=weight, ha="center", va="center", zorder=3)

        # active-site: magma underline just BELOW the cell + dot below that
        if pos in active:
            ax.plot([x0 + 0.01, x0 + cell_w - 0.01], [y0 - 0.02, y0 - 0.02],
                    color=COL_ACTIVE, linewidth=1.3, solid_capstyle="butt", zorder=4)
            ax.plot([x0 + cell_w / 2], [y0 - 0.10], marker="o",
                    markersize=1.6, color=COL_ACTIVE, zorder=4)

        # interface (§16b): teal overline just ABOVE the cell + dot above that
        if pos in iface:
            top = y0 + cell_h + 0.02
            ax.plot([x0 + 0.01, x0 + cell_w - 0.01], [top, top],
                    color=COL_INTERFACE, linewidth=1.3, solid_capstyle="butt", zorder=4)
            ax.plot([x0 + cell_w / 2], [top + 0.08], marker="o",
                    markersize=1.6, color=COL_INTERFACE, zorder=4)

        # mutation outline (orange, or magma if AS-mutated, teal if iface-mutated)
        if pos in mut_map:
            edge = COL_MUTATION
            if pos in active:
                edge = COL_MUT_AS
            elif pos in iface:
                edge = COL_MUT_IF
            ax.add_patch(Rectangle((x0, y0), cell_w, cell_h, facecolor="none",
                                   edgecolor=edge, linewidth=0.9, zorder=5))

    # row-start position labels in the gutter
    for row in range(n_rows):
        row_start_pos = row * wrap + 1
        yc = fig_h - top_pad - (row + 1) * row_h_in + 0.15 + cell_h / 2
        ax.text(left_gutter - 0.05, yc, str(row_start_pos),
                fontsize=6.5, color="#555", ha="right", va="center")

    # legend — measure real text extents so labels never collide
    lg_y = 0.10
    entries = [
        ("conservation (white \u2192 magma)", None, "gradient"),
        ("active site", COL_ACTIVE, "underline"),
        ("interface (\u00a716b)", COL_INTERFACE, "overline"),
        ("mutation", COL_MUTATION, "outline"),
    ]
    sw_w = 0.18
    sw_h = 0.14
    sw_label_gap = 0.06   # inches between swatch and label
    entry_gap    = 0.28   # inches between entries
    lg_font      = 7.0

    # renderer for measuring text width in inches
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi_ = fig.dpi

    x = left_gutter
    for label, col, kind in entries:
        if kind == "gradient":
            steps = 12
            for si in range(steps):
                frac = si / max(1, steps - 1)
                ax.add_patch(Rectangle((x + si * sw_w / steps, lg_y),
                                       sw_w / steps, sw_h,
                                       facecolor=_cons_bg(frac),
                                       edgecolor="none", zorder=1))
        elif kind == "underline":
            ax.plot([x, x + sw_w], [lg_y, lg_y], color=col, linewidth=1.5, zorder=2)
        elif kind == "overline":
            ax.plot([x, x + sw_w], [lg_y + sw_h, lg_y + sw_h], color=col, linewidth=1.5, zorder=2)
        elif kind == "outline":
            ax.add_patch(Rectangle((x, lg_y), sw_w, sw_h, facecolor="none",
                                   edgecolor=col, linewidth=1.1, zorder=2))
        label_x = x + sw_w + sw_label_gap
        t = ax.text(label_x, lg_y + sw_h / 2, label, fontsize=lg_font,
                    va="center", ha="left", color="#333")
        # measured width in inches (bbox is in display pixels)
        bbox = t.get_window_extent(renderer=renderer)
        label_w_in = bbox.width / dpi_
        x = label_x + label_w_in + entry_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.05)
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), format="png",
                    bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)


# ---- CLI --------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", type=Path,
                     help="assembled results.json (webapp bundle)")
    src.add_argument("--candidates", type=Path, nargs="+",
                     help="one or more candidates.json files (11_generate.py output)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--wrap", type=int, default=60,
                    help="residues per row (default 60)")
    ap.add_argument("--png", action="store_true",
                    help="also write a sibling .png at --dpi")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title-prefix", default="",
                    help="prepended to per-design title")
    args = ap.parse_args(argv)

    tracks: list[tuple[str, str, dict]] = []
    if args.results:
        r = json.loads(args.results.read_text())
        tracks = _tracks_from_results(r)
        if not tracks:
            print(f"[render_track_svg] no tracks in {args.results}", file=sys.stderr)
            return 2
    else:
        for cp in args.candidates:
            c = json.loads(cp.read_text())
            tracks.extend(_tracks_from_candidates(c))

    written = 0
    for ph, did, track in tracks:
        out = args.out_dir / f"{ph}_{did}.svg"
        n_mut = len(track.get("mutations", []))
        n_iface = len(track.get("interfaces", []))
        n_as = len(track.get("active_site", []))
        title = (f"{args.title_prefix}{ph} \u00b7 {did} \u00b7 "
                 f"{n_mut} mut \u00b7 {n_as} active-site \u00b7 {n_iface} interface (\u00a716b)")
        render_track(track, out, wrap=args.wrap, dpi=args.dpi,
                     also_png=args.png, title=title)
        written += 1
    print(f"[render_track_svg] wrote {written} track(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
