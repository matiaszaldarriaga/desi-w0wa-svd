#!/usr/bin/env python3
"""Build the claim provenance dashboard from the YAML registries and paper data.

The dashboard (``output/claim_dashboard.html``) is a *single, self-contained*
HTML file aimed at two audiences: human readers and the AI agents that crawl the
public repository. It lets either navigate the paper's logic — the claim
dependency graph, and, per claim, the evidence, supporting figures, quoted
numbers, and links to the exact code that produced them.

Design notes
------------
* **Self-contained.** Figures are rasterized and embedded as base64 data URIs,
  so the file renders anywhere with no external ``dashboard_assets/`` directory
  and no display-time image conversion. Copy the one file and it just works.
* **Layout-independent links.** Source/section/number links point at the public
  GitHub repository (``GITHUB_BASE``), with a path rewrite from the working-tree
  layout to the flattened release layout, so they resolve no matter where the
  HTML file lives.
* **Navigation, not audit.** The shipped page is reader/bot-facing navigation
  plus one positive provenance signal. The internal quality-control checks
  (provenance completeness, number mismatches, missing scripts, unlinked
  figures) run at *build time* — printed to the console and written to an
  author-only report (``output/dashboard_audit.md``) that is NOT shipped.

This script is the single source of truth for the dashboard; the release build
process must run it so the published dashboard is regenerated from the current
registries (see docs/superpowers/plans/2026-03-31-release-directory.md).

Usage:
    python scripts/build_dashboard.py
"""

import base64
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Public repository the shipped links resolve against.
GITHUB_BASE = "https://github.com/matiaszaldarriaga/desi-w0wa-svd/blob/main/"


# ── Working-tree → public-repo path mapping ──────────────────────────────────
#
# The working tree uses calculation/scripts, code/figures, input/reference_data;
# the public (release) repo flattens these. Links must point at the public
# layout so they resolve in the shipped artifact.


def to_repo_path(p):
    """Map a working-tree path to its location in the public (flat) repo."""
    p = p.replace("calculation/scripts/", "scripts/")
    p = p.replace("code/figures/", "scripts/")
    p = p.replace("input/reference_data/", "data/")
    return p


def gh_url(p, line=None):
    """GitHub blob URL for a working-tree path, optionally with a line anchor."""
    url = GITHUB_BASE + to_repo_path(p)
    if line:
        url += f"#L{line}"
    return url


# Literature evidence is stored under input/literature/<arxiv-id>/; it is a
# citation, not code, and is rendered as an arXiv link rather than a repo link.
_ARXIV_LIT = re.compile(r"input/literature/(\d{4}\.\d{4,5})/")


def is_literature_ref(ref):
    return bool(_ARXIV_LIT.search(ref)) or ref.endswith(".pdf")


def classify_ref(ref):
    """Return ('code'|'reference', html_link) for an evidence data_ref."""
    m = _ARXIV_LIT.search(ref)
    if m:
        aid = m.group(1)
        return "reference", f'<a href="https://arxiv.org/abs/{aid}">arXiv:{aid}</a>'
    return "code", f'<a href="{gh_url(ref)}">{_esc(Path(ref).name)}</a>'


# ── Data loaders ─────────────────────────────────────────────────────────────


def load_claims():
    with open(PROJECT_ROOT / "structure" / "claims.yaml") as f:
        return yaml.safe_load(f)["claims"]


def load_figures():
    with open(PROJECT_ROOT / "structure" / "figures.yaml") as f:
        return yaml.safe_load(f)["figures"]


def load_scripts():
    with open(PROJECT_ROOT / "structure" / "scripts.yaml") as f:
        return yaml.safe_load(f)["scripts"]


def load_numbers():
    # Layout-aware: working tree keeps it under input/reference_data/, the
    # flattened release under data/.
    for rel in ("input/reference_data/paper_numbers.json", "data/paper_numbers.json"):
        p = PROJECT_ROOT / rel
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError("paper_numbers.json not found in input/reference_data/ or data/")


# ── LaTeX context extraction ─────────────────────────────────────────────────


def extract_claim_contexts():
    """Find all \\claim{id}{...} in .tex files and return context per claim_id."""
    contexts = {}
    sections_dir = PROJECT_ROOT / "paper" / "sections"
    for tex_file in sorted(sections_dir.glob("*.tex")):
        text = tex_file.read_text()
        for m in re.finditer(r"\\claim\{([^}]+)\}\{", text):
            claim_id = m.group(1)
            # Balanced-brace match for the second argument.
            start_brace = m.end() - 1
            depth = 0
            end_brace = start_brace
            for i in range(start_brace, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end_brace = i
                        break
            statement = text[start_brace + 1 : end_brace]
            line_num = text[: m.start()].count("\n") + 1
            entry = {"file": tex_file.name, "line": line_num, "statement": statement}
            if claim_id in contexts:
                if isinstance(contexts[claim_id], list):
                    contexts[claim_id].append(entry)
                else:
                    contexts[claim_id] = [contexts[claim_id], entry]
            else:
                contexts[claim_id] = entry
    return contexts


# ── Figure rasterization → base64 data URIs ──────────────────────────────────


def _rasterize_pdf(pdf_path, out_png):
    """Rasterize the first page of a PDF to PNG using the first available backend.

    Tries pdftoppm (poppler, cross-platform) then sips (macOS). Returns True on
    success. Keeping multiple backends means the build is not macOS-only.
    """
    out_png = Path(out_png)
    prefix = out_png.with_suffix("")  # pdftoppm appends ".png"
    attempts = [
        ["pdftoppm", "-png", "-r", "110", "-singlefile", str(pdf_path), str(prefix)],
        ["sips", "-s", "format", "png", "-Z", "1400", str(pdf_path), "--out", str(out_png)],
    ]
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if out_png.exists():
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def build_figure_data_uris(figures_yaml):
    """Rasterize every registered figure PDF and return {png_name: data_uri}.

    Rasterization happens in a temporary directory; nothing persistent is
    written, so the script behaves identically in the working-tree and flattened
    release layouts. Images are embedded as base64 (no external assets dir).
    """
    pdf_paths = []
    for fig in figures_yaml:
        for ff in fig.get("file", "").split(","):
            ff = ff.strip()
            if ff:
                pdf_paths.append(PROJECT_ROOT / ff)

    uris = {}
    stats = {"converted": 0, "failed": [], "missing": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for pdf in sorted(set(pdf_paths)):
            png_name = pdf.with_suffix(".png").name
            if not pdf.exists():
                stats["missing"].append(str(pdf.relative_to(PROJECT_ROOT)))
                continue
            png = tmp_dir / png_name
            if _rasterize_pdf(pdf, png):
                stats["converted"] += 1
            else:
                stats["failed"].append(pdf.name)
                continue
            data = base64.b64encode(png.read_bytes()).decode("ascii")
            uris[png_name] = f"data:image/png;base64,{data}"
    return uris, stats


# ── Number matching (build-time audit only) ──────────────────────────────────

# For each claim, the paper_numbers.json keys whose values it quotes. The
# dashboard displays the value resolved live from paper_numbers.json — the
# canonical source — so a printed number can never drift from the registry, and
# a renamed/removed key surfaces in the build self-check rather than silently.
# (Deeper paper-text vs JSON verification is the job of verify_paper_numbers.py.)
CLAIM_NUMBER_KEYS = {
    "c0_universal": [
        "section3_inner_products.min",
        "section3_inner_products.mean",
    ],
    "c0_is_omegamh2": [
        "section3_sequential_R2.omh2_only",
        "section3_sequential_R2.plus_ombh2",
        "section3_sequential_R2.plus_theta",
        "eq6_c0_formula.beta_omh2",
        "eq6_c0_formula.beta_ombh2",
        "eq6_c0_formula.beta_theta",
    ],
    "c0_tensions_sign": [
        "table5_c0_tensions.BAO_ACT.tension",
        "table5_c0_tensions.Union3_ACT.tension",
        "table5_c0_tensions.Pantheon+_ACT.tension",
        "table5_c0_tensions.DES-Dovekie_ACT.tension",
    ],
    "bao_constrains_omh2": [
        "table3_sigma_c.BAO.sigma_c0",
        "table3_sigma_c.Union3.sigma_c0",
        "table3_sigma_c.Pantheon+.sigma_c0",
        "table3_sigma_c.DES-Dovekie.sigma_c0",
    ],
    "w0wa_is_c0": [
        "table9_tensions.BAO.c1_tension",
        "table5_c0_tensions.BAO_ACT.tension",
    ],
    "c0_dominant_w0wa": [
        "table6_grid_ranges.BAO.c0",
        "table6_grid_ranges.BAO.c1",
    ],
    "freed_calpha_pattern": [],
    "three_mode_ladder": [
        "pivot_fits.c1_BAO.z_pivot",
        "pivot_fits.c1_BAO.w_data",
        "pivot_fits.c1_BAO.sigma_w_meas",
    ],
    "only_omk_measurable": [
        "omk_fits.sigma_res",
    ],
    "omk_coherence": [
        "omk_coherence.c0.implied_Omk",
        "omk_coherence.c1.implied_Omk",
    ],
    "sn_blind_curvature": [
        "table13_new_directions.Omk_Union3.sigma_res",
        "table13_new_directions.Omk_Pantheon+.sigma_res",
        "table13_new_directions.Omk_DES-Dovekie.sigma_res",
    ],
    "alens_dilutes": [
        "table12_ext_c0.Alens.tension",
        "table12_ext_c0.LCDM.tension",
    ],
}


def resolve_json_path(numbers, dotted_key):
    """Resolve a dotted key like 'table5_c0_tensions.BAO_ACT.tension'."""
    obj = numbers
    for part in dotted_key.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def _fmt_number(v):
    """Format a JSON number for display (trim trailing zeros, keep signs)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    s = f"{v:.5g}"
    return s


def get_claim_numbers(claim_id, numbers):
    """Return [{key, value}] for a claim, with values from paper_numbers.json.

    ``value`` is None when the key is missing (flagged by the build self-check).
    """
    results = []
    for key in CLAIM_NUMBER_KEYS.get(claim_id, []):
        results.append({"key": key, "value": resolve_json_path(numbers, key)})
    return results


def check_provenance(claim):
    """Audit-only provenance grade: 'green' | 'yellow' | 'red'."""
    issues = []
    evidence_list = claim.get("evidence", [])
    if not evidence_list:
        issues.append("no evidence items in YAML")
    for ev in evidence_list:
        refs = ev.get("data_refs", [])
        if not refs:
            issues.append(f"evidence '{ev.get('id')}' has no data_refs")
        for ref in refs:
            if is_literature_ref(ref):
                continue  # bibliographic citation, not shipped code
            # Resolve against either layout (working tree path or flattened repo path).
            if not (
                (PROJECT_ROOT / ref).exists()
                or (PROJECT_ROOT / to_repo_path(ref)).exists()
            ):
                issues.append(f"missing script: {ref}")
    if not claim.get("_context"):
        issues.append("no \\claim annotation found in LaTeX")
    if not issues:
        return "green", issues
    if len(issues) <= 1:
        return "yellow", issues
    return "red", issues


def run_build_audit(claims, figures_yaml, numbers, fig_stats):
    """Run author-facing QC checks; print to console and write an author report.

    Returns ``all_clean`` (bool) used to set the dashboard's provenance banner.
    Nothing here is shipped in the HTML.
    """
    grades = {}
    issues_by_claim = {}
    for claim in claims:
        grade, issues = check_provenance(claim)
        grades[claim["id"]] = grade
        if issues:
            issues_by_claim[claim["id"]] = issues

    n_green = sum(1 for g in grades.values() if g == "green")
    n_yellow = sum(1 for g in grades.values() if g == "yellow")
    n_red = sum(1 for g in grades.values() if g == "red")

    missing_keys = []
    for claim in claims:
        for entry in get_claim_numbers(claim["id"], numbers):
            if entry["value"] is None:
                missing_keys.append((claim["id"], entry["key"]))

    unlinked = [f.get("label", "?") for f in figures_yaml if not f.get("supports_claims")]

    all_clean = (
        n_yellow == 0
        and n_red == 0
        and not missing_keys
        and not fig_stats["failed"]
        and not fig_stats["missing"]
    )

    # ── Author-only report (not shipped) ──
    report = [
        "# Dashboard build self-check",
        "",
        f"Generated {date.today().isoformat()} by `scripts/build_dashboard.py`.",
        "This is an author-facing audit. It is **not** part of the shipped dashboard.",
        "",
        "## Provenance",
        f"- Claims fully traced (code + data + LaTeX annotation): "
        f"**{n_green}/{len(claims)}**",
    ]
    if n_yellow or n_red:
        report.append(f"- Partial: {n_yellow}, Missing: {n_red}")
        for cid, issues in issues_by_claim.items():
            report.append(f"  - `{cid}`: {'; '.join(issues)}")
    report.append("")
    report.append("## Quoted-number provenance keys")
    if missing_keys:
        report.append(
            f"- **{len(missing_keys)} key(s) not found** in paper_numbers.json "
            "(renamed/removed — fix the mapping in build_dashboard.py):"
        )
        for cid, key in missing_keys:
            report.append(f"  - `{cid}` -> `{key}`")
    else:
        report.append("- All claim number keys resolve in `paper_numbers.json`. ✓")
    report.append("")
    report.append("## Figures")
    report.append(f"- Rasterized and embedded: {fig_stats['converted']}")
    if fig_stats["failed"]:
        report.append(f"- **Failed to rasterize:** {', '.join(fig_stats['failed'])}")
    if fig_stats["missing"]:
        report.append(f"- **Missing PDF(s):** {', '.join(fig_stats['missing'])}")
    if unlinked:
        report.append(
            f"- Not tied to a specific claim (shown as 'Overview figures'): "
            f"{', '.join(unlinked)}"
        )
    report.append("")
    report.append(f"## Verdict\n\n{'All checks pass. ✓' if all_clean else 'See issues above.'}")

    report_dir = PROJECT_ROOT / "output" if (PROJECT_ROOT / "output").is_dir() else PROJECT_ROOT
    report_path = report_dir / "dashboard_audit.md"
    report_path.write_text("\n".join(report) + "\n")

    # ── Console summary ──
    print("Build self-check (author-only, not shipped):")
    print(f"  Provenance: {n_green}/{len(claims)} fully traced", end="")
    print(f" ({n_yellow} partial, {n_red} missing)" if (n_yellow or n_red) else "")
    if missing_keys:
        print(f"  ⚠ {len(missing_keys)} number key(s) not found in paper_numbers.json")
    else:
        print("  Numbers: all claim keys resolve in paper_numbers.json")
    if fig_stats["failed"]:
        print(f"  ⚠ figures failed to rasterize: {', '.join(fig_stats['failed'])}")
    if fig_stats["missing"]:
        print(f"  ⚠ missing figure PDFs: {', '.join(fig_stats['missing'])}")
    if unlinked:
        print(f"  Overview figures (no single claim): {', '.join(unlinked)}")
    print(f"  Report: {report_path}")
    if not all_clean:
        print("  WARNING: build self-check found issues — see report above.")
    print()
    return all_clean


# ── DAG layout (topological) ─────────────────────────────────────────────────


def compute_dag_layout(claims):
    """Compute (x, y) positions for DAG nodes using a layered layout."""
    claim_ids = [c["id"] for c in claims]
    depends = {c["id"]: c.get("depends_on", []) for c in claims}
    layers = {}

    def get_layer(cid, visited=None):
        if visited is None:
            visited = set()
        if cid in layers:
            return layers[cid]
        if cid in visited:
            return 0
        visited.add(cid)
        deps = depends.get(cid, [])
        if not deps:
            layers[cid] = 0
            return 0
        layers[cid] = max(get_layer(d, visited) for d in deps if d in depends) + 1
        return layers[cid]

    for cid in claim_ids:
        get_layer(cid)

    layer_groups = defaultdict(list)
    for cid, layer in layers.items():
        layer_groups[layer].append(cid)

    node_w, node_h, x_gap, y_gap = 160, 60, 30, 100
    max_in_layer = max(len(v) for v in layer_groups.values())
    total_width = max_in_layer * (node_w + x_gap)
    positions = {}
    for layer_idx in sorted(layer_groups):
        nodes = layer_groups[layer_idx]
        layer_width = len(nodes) * node_w + (len(nodes) - 1) * x_gap
        start_x = (total_width - layer_width) / 2
        for i, cid in enumerate(nodes):
            positions[cid] = (start_x + i * (node_w + x_gap), layer_idx * (node_h + y_gap))
    return positions, node_w, node_h


SHORT_LABELS = {
    "c0_universal": "c0 universal",
    "c0_is_omegamh2": "c0 = Omh2",
    "c0_tensions_sign": "c0 tensions",
    "bao_constrains_omh2": "BAO constrains",
    "w0wa_is_c0": "w0wa = c0",
    "c0_dominant_w0wa": "c0 dominant",
    "freed_calpha_pattern": "freed calpha",
    "three_mode_ladder": "pivot w=-1",
    "only_omk_measurable": "only Omk new",
    "omk_coherence": "Omk coherence",
    "sn_blind_curvature": "SN blind curv",
    "alens_dilutes": "Alens dilutes",
    "tau_single_point_failure": "tau systematic",
}


def generate_dag_svg(claims):
    """Generate an SVG of the claim dependency graph (uniform, neutral styling)."""
    positions, node_w, node_h = compute_dag_layout(claims)
    depends = {c["id"]: c.get("depends_on", []) for c in claims}

    section_map = {}
    for c in claims:
        sec = c.get("section", "")
        m = re.match(r"([\S]+\d+)", sec)
        section_map[c["id"]] = m.group(1) if m else (sec.split()[0] if sec else "")

    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    svg_w = max(all_x) + node_w + 40 if all_x else 800
    svg_h = max(all_y) + node_h + 40 if all_y else 400

    lines = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" width="100%" '
        f'style="max-width:{int(svg_w)}px; display:block; margin:0 auto;">',
        "  <defs>",
        '    <marker id="arrowhead" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">',
        '      <polygon points="0 0, 10 3.5, 0 7" fill="#9aa0b4"/>',
        "    </marker>",
        "  </defs>",
    ]
    for cid, deps in depends.items():
        if cid not in positions:
            continue
        cx, cy = positions[cid]
        for dep in deps:
            if dep not in positions:
                continue
            dx, dy = positions[dep]
            x1, y1 = dx + node_w / 2, dy + node_h
            x2, y2 = cx + node_w / 2, cy
            mid_y = (y1 + y2) / 2
            lines.append(
                f'  <path d="M {x1},{y1} C {x1},{mid_y} {x2},{mid_y} {x2},{y2}" '
                f'fill="none" stroke="#b8bcd0" stroke-width="1.5" marker-end="url(#arrowhead)"/>'
            )
    for c in claims:
        cid = c["id"]
        if cid not in positions:
            continue
        x, y = positions[cid]
        label = SHORT_LABELS.get(cid, cid)
        sec = section_map.get(cid, "")
        lines.append(
            f'  <g class="dag-node" onclick="scrollToPanel(\'{cid}\')" style="cursor:pointer">'
        )
        lines.append(
            f'    <rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="8" ry="8" '
            f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>'
        )
        lines.append(
            f'    <text x="{x + node_w/2}" y="{y + 24}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="#1e1b4b">{_esc(label)}</text>'
        )
        lines.append(
            f'    <text x="{x + node_w/2}" y="{y + 42}" text-anchor="middle" '
            f'font-size="10" fill="#6b7280">{_esc(sec)}</text>'
        )
        lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines)


def _esc(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _clean_latex(s):
    """Strip common LaTeX commands for plain-text display in HTML."""
    repl = {
        "\\czero": "c0", "\\cone": "c1", "\\omh": "Omega_m h^2",
        "\\obh": "Omega_b h^2", "\\thetastar": "theta*", "\\Omk": "Omega_k",
        "\\wowa": "w0wa", "\\LCDM": "LCDM", "\\sigres": "sigma_res",
        "\\mathrm{lens}": "lens", "\\sim": "~", "\\pm": "+/-",
        "\\sigma": "sigma", "\\approx": "~",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\$([^$]*)\$", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"\{|\}", "", s)
    return s.strip()


# ── HTML generation ──────────────────────────────────────────────────────────


def generate_detail_panel(claim, figures_by_claim, numbers, fig_uris):
    """Generate the navigation panel for a single claim (reader/bot-facing)."""
    cid = claim["id"]
    statement = claim.get("statement", "")
    section = claim.get("section", "")
    ctx = claim.get("_context")

    lines = [
        f'<div id="panel-{cid}" class="claim-panel">',
        f'  <div class="panel-header" onclick="togglePanel(\'{cid}\')">',
        f"    <strong>{_esc(cid)}</strong>",
        f'    <span class="section-tag">{_esc(section)}</span>',
        f'    <span class="expand-icon" id="icon-{cid}">+</span>',
        "  </div>",
        f'  <p class="claim-statement">{_esc(_clean_latex(statement))}</p>',
        f'  <div id="detail-{cid}" class="panel-detail" style="display:none">',
    ]

    # Location (link to the LaTeX source on GitHub).
    ctx_entries = ctx if isinstance(ctx, list) else ([ctx] if ctx else [])
    if ctx_entries:
        lines.append("    <h4>Defined in</h4>")
        for c in ctx_entries:
            url = gh_url(f"paper/sections/{c['file']}", c["line"])
            lines.append(f'    <p><a href="{url}">{_esc(c["file"])}:{c["line"]}</a></p>')

    # Evidence (description + GitHub links to the producing code).
    evidence_list = claim.get("evidence", [])
    if evidence_list:
        lines.append("    <h4>Evidence</h4>")
        lines.append("    <ul>")
        for ev in evidence_list:
            lines.append("      <li>")
            lines.append(
                f"        <strong>{_esc(ev.get('id',''))}</strong>: "
                f"{_esc(ev.get('description',''))}"
            )
            code_links, ref_links = [], []
            for ref in ev.get("data_refs", []):
                kind, html = classify_ref(ref)
                (code_links if kind == "code" else ref_links).append(html)
            if code_links:
                lines.append(f"        <br>Code: {', '.join(code_links)}")
            if ref_links:
                lines.append(f"        <br>Reference: {', '.join(ref_links)}")
            lines.append("      </li>")
        lines.append("    </ul>")

    # Figures (embedded base64).
    figs = figures_by_claim.get(cid, [])
    if figs:
        lines.append("    <h4>Figures</h4>")
        lines.append('    <div class="figures-grid">')
        for fig in figs:
            for ff in fig.get("file", "").split(","):
                ff = ff.strip()
                if not ff:
                    continue
                png_name = Path(ff).stem + ".png"
                uri = fig_uris.get(png_name)
                label = fig.get("label", "")
                lines.append('      <div class="fig-card">')
                if uri:
                    lines.append(
                        f'        <img src="{uri}" alt="{_esc(label)}" loading="lazy">'
                    )
                lines.append(f"        <p>{_esc(label)}</p>")
                lines.append("      </div>")
        lines.append("    </div>")

    # Numbers (provenance: the canonical value from paper_numbers.json + its key).
    num_entries = [e for e in get_claim_numbers(cid, numbers) if e["value"] is not None]
    if num_entries:
        json_url = gh_url("input/reference_data/paper_numbers.json")
        lines.append("    <h4>Key numbers</h4>")
        lines.append('    <table class="numbers-table">')
        lines.append("      <tr><th>Value</th><th>Traces to (paper_numbers.json)</th></tr>")
        for entry in num_entries:
            lines.append(
                "      <tr>"
                f"<td>{_esc(_fmt_number(entry['value']))}</td>"
                f'<td><a href="{json_url}"><code>{_esc(entry["key"])}</code></a></td>'
                "</tr>"
            )
        lines.append("    </table>")

    # Dependencies / dependents.
    deps = claim.get("depends_on", [])
    if deps:
        dep_links = [
            f'<a href="#panel-{d}" onclick="scrollToPanel(\'{d}\')">{_esc(d)}</a>'
            for d in deps
        ]
        lines.append(f"    <p class=\"deps\">Depends on: {', '.join(dep_links)}</p>")
    dependents = claim.get("_dependents", [])
    if dependents:
        dep_links = [
            f'<a href="#panel-{d}" onclick="scrollToPanel(\'{d}\')">{_esc(d)}</a>'
            for d in dependents
        ]
        lines.append(f"    <p class=\"deps\">Required by: {', '.join(dep_links)}</p>")

    lines.append("  </div>")
    lines.append("</div>")
    return "\n".join(lines)


def generate_provenance_banner(claims, figures_yaml, all_clean):
    """One positive, truthful provenance line for the public page."""
    n_claims = len(claims)
    n_figs = len(figures_yaml)
    traced = (
        " &middot; <span class=\"check\">every claim traces to code, data &amp; figures &#10003;</span>"
        if all_clean
        else ""
    )
    return f"""
    <div class="banner">
      <strong>{n_claims} claims</strong> &middot; <strong>{n_figs} figures</strong>{traced}
      <div class="banner-sub">Click any claim to open its evidence, figures, quoted numbers, and the code that produced them.</div>
    </div>
    """


def generate_overview_figures(figures_yaml, fig_uris):
    """Figures that introduce the data rather than support a single claim."""
    overview = [f for f in figures_yaml if not f.get("supports_claims")]
    if not overview:
        return ""
    lines = [
        '<div class="overview-section">',
        "  <h2>Overview figures</h2>",
        "  <p>These figures introduce the datasets and distance measurements used "
        "throughout the paper. They support the analysis as a whole rather than a "
        "single claim.</p>",
        '  <div class="figures-grid">',
    ]
    for fig in overview:
        label = fig.get("label", "unknown")
        section = fig.get("section", "")
        for ff in fig.get("file", "").split(","):
            ff = ff.strip()
            if not ff:
                continue
            png_name = Path(ff).stem + ".png"
            uri = fig_uris.get(png_name)
            lines.append('    <div class="fig-card">')
            if uri:
                lines.append(f'      <img src="{uri}" alt="{_esc(label)}" loading="lazy">')
            lines.append(f"      <p>{_esc(label)} ({_esc(section)})</p>")
            lines.append("    </div>")
    lines.append("  </div>")
    lines.append("</div>")
    return "\n".join(lines)


CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  max-width: 1100px; margin: 0 auto; padding: 24px;
  background: #fafafb; color: #2b2b33; line-height: 1.55;
}
h1 { margin-bottom: 2px; }
.subtitle { color: #6b7280; margin-top: 0; margin-bottom: 20px; }
.intro { background:#fff; border:1px solid #e6e6ea; border-radius:8px;
  padding:14px 18px; margin-bottom:20px; font-size:14px; color:#444; }

.banner {
  background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px;
  padding: 14px 20px; margin-bottom: 22px; font-size: 16px;
}
.banner .check { color: #4338ca; }
.banner-sub { font-size: 13px; color: #6b7280; margin-top: 4px; }

.dag-section {
  background: #fff; border: 1px solid #e6e6ea; border-radius: 8px;
  padding: 20px; margin-bottom: 24px; overflow-x: auto;
}
.dag-section h2 { margin-top: 0; }
.dag-node:hover rect { stroke-width: 2.5; }

.claim-panel {
  border: 1px solid #e6e6ea; border-left: 4px solid #6366f1; border-radius: 8px;
  margin: 8px 0; background: #fff;
}
.claim-panel:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.panel-header {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px; cursor: pointer; user-select: none;
}
.panel-header:hover { background: #f7f7fb; }
.section-tag {
  margin-left: auto; font-size: 12px; color: #6b7280;
  background: #f0f0f4; padding: 2px 8px; border-radius: 4px;
}
.expand-icon { font-size: 18px; font-weight: 300; color: #9aa0b4; width: 24px; text-align: center; }
.claim-statement { padding: 0 16px 10px 16px; margin: 0; font-size: 14px; color: #555; }
.panel-detail { padding: 0 16px 16px 16px; border-top: 1px solid #eee; }
.panel-detail h4 {
  margin: 16px 0 8px 0; font-size: 12px; text-transform: uppercase;
  letter-spacing: 0.5px; color: #8a8f9c;
}
.deps { font-size: 13px; color: #555; }

.figures-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.fig-card {
  border: 1px solid #e6e6ea; border-radius: 6px; padding: 8px;
  background: #fafafb; max-width: 380px;
}
.fig-card img { max-width: 100%; height: auto; border-radius: 4px; }
.fig-card p { margin: 6px 0 0 0; font-size: 12px; color: #6b7280; text-align: center; }

.numbers-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.numbers-table th {
  background: #f5f5f8; padding: 6px 10px; text-align: left;
  border-bottom: 2px solid #ddd; font-weight: 600;
}
.numbers-table td { padding: 5px 10px; border-bottom: 1px solid #eee; }
.numbers-table code { font-size: 12px; background: #f0f0f4; padding: 1px 4px; border-radius: 3px; }

a { color: #4f46e5; text-decoration: none; }
a:hover { text-decoration: underline; }
ul { padding-left: 20px; }
li { margin-bottom: 6px; }

.overview-section {
  background: #fff; border: 1px solid #e6e6ea; border-radius: 8px;
  padding: 20px; margin-top: 24px;
}
.overview-section h2 { margin-top: 0; }

.highlight { animation: flash 1s ease-out; }
@keyframes flash { 0% { background: #faf5d7; } 100% { background: transparent; } }

.footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #eee;
  font-size: 12px; color: #9aa0b4; text-align: center; }
"""

JS = """
function togglePanel(id) {
  var detail = document.getElementById('detail-' + id);
  var icon = document.getElementById('icon-' + id);
  if (detail.style.display === 'none') {
    detail.style.display = 'block';
    if (icon) icon.textContent = String.fromCharCode(8722);
  } else {
    detail.style.display = 'none';
    if (icon) icon.textContent = '+';
  }
}
function scrollToPanel(id) {
  var panel = document.getElementById('panel-' + id);
  if (!panel) return;
  var detail = document.getElementById('detail-' + id);
  var icon = document.getElementById('icon-' + id);
  if (detail && detail.style.display === 'none') {
    detail.style.display = 'block';
    if (icon) icon.textContent = String.fromCharCode(8722);
  }
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  panel.classList.add('highlight');
  setTimeout(function() { panel.classList.remove('highlight'); }, 1200);
}
function expandAll() {
  document.querySelectorAll('.panel-detail').forEach(function(el){ el.style.display='block'; });
  document.querySelectorAll('.expand-icon').forEach(function(el){ el.textContent=String.fromCharCode(8722); });
}
function collapseAll() {
  document.querySelectorAll('.panel-detail').forEach(function(el){ el.style.display='none'; });
  document.querySelectorAll('.expand-icon').forEach(function(el){ el.textContent='+'; });
}
"""


def generate_html(claims, figures_yaml, numbers, fig_uris, all_clean):
    """Assemble the complete self-contained dashboard HTML."""
    figures_by_claim = defaultdict(list)
    for fig in figures_yaml:
        for cid in fig.get("supports_claims", []):
            figures_by_claim[cid].append(fig)

    # Reverse dependency edges.
    for c in claims:
        c["_dependents"] = []
    claim_map = {c["id"]: c for c in claims}
    for c in claims:
        for dep in c.get("depends_on", []):
            if dep in claim_map:
                claim_map[dep]["_dependents"].append(c["id"])

    dag_svg = generate_dag_svg(claims)
    banner = generate_provenance_banner(claims, figures_yaml, all_clean)
    overview = generate_overview_figures(figures_yaml, fig_uris)

    section_order = [
        "1", "intro", "2", "data", "3", "universal", "4", "tension",
        "5", "w0wa", "reinterpretation", "6", "curvature", "7", "extension",
        "8", "conclusion", "App", "appendix",
    ]

    def section_sort_key(claim):
        sec = claim.get("section", "").lower()
        for i, kw in enumerate(section_order):
            if kw.lower() in sec:
                return i
        return 999

    panels = "".join(
        generate_detail_panel(c, figures_by_claim, numbers, fig_uris)
        for c in sorted(claims, key=section_sort_key)
    )

    repo_link = GITHUB_BASE.rsplit("/blob/", 1)[0]
    generated = date.today().isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claim Provenance Dashboard -- DESI-w0wa</title>
<style>
{CSS}
</style>
</head>
<body>

<h1>Claim Provenance Dashboard</h1>
<p class="subtitle">Universal distance modes from DESI BAO and Type Ia supernovae</p>

<div class="intro">
  This page maps the paper's argument to its evidence. Each scientific claim is
  shown with the figures, quoted numbers, and code that support it, and with the
  other claims it depends on. It is meant to be navigated by readers and by AI
  agents alike: every number traces to a key in <code>paper_numbers.json</code>,
  and every figure and script links to its source in the
  <a href="{repo_link}">public repository</a>.
</div>

{banner}

<div class="dag-section">
  <h2>Claim dependency graph</h2>
  <p style="font-size:13px; color:#6b7280; margin-top:-8px;">
    Arrows point from a claim to the claims that build on it. Click any node to jump to its detail panel.
  </p>
  {dag_svg}
</div>

<div style="display:flex; gap:8px; margin-bottom:12px;">
  <button onclick="expandAll()" style="padding:6px 14px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:13px;">Expand all</button>
  <button onclick="collapseAll()" style="padding:6px 14px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:13px;">Collapse all</button>
</div>

<h2>Claims</h2>

{panels}

{overview}

<div class="footer">
  Generated {generated} from the claim, figure, and number registries
  (<code>structure/*.yaml</code>) by <code>scripts/build_dashboard.py</code>.
  Source: <a href="{repo_link}">{repo_link}</a>
</div>

<script>
{JS}
</script>

</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("Building claim provenance dashboard...\n")

    print("Loading registries...")
    claims = load_claims()
    figures = load_figures()
    numbers = load_numbers()
    print(f"  Claims: {len(claims)}  Figures: {len(figures)}  Number keys: {len(numbers)}\n")

    print("Extracting LaTeX claim annotations...")
    contexts = extract_claim_contexts()
    for claim in claims:
        claim["_context"] = contexts.get(claim["id"])
    print(f"  Found {len(contexts)} \\claim annotations\n")

    print("Embedding figures (base64)...")
    fig_uris, fig_stats = build_figure_data_uris(figures)
    print(f"  {len(fig_uris)} figures embedded\n")

    all_clean = run_build_audit(claims, figures, numbers, fig_stats)

    print("Generating HTML...")
    html = generate_html(claims, figures, numbers, fig_uris, all_clean)
    # Output location adapts to layout: working tree -> output/, flattened
    # release (no output/ dir) -> structure/ alongside the registries.
    if (PROJECT_ROOT / "output").is_dir():
        out_path = PROJECT_ROOT / "output" / "claim_dashboard.html"
    else:
        out_path = PROJECT_ROOT / "structure" / "claim_dashboard.html"
    out_path.write_text(html)
    print(f"  Written to: {out_path}")
    print(f"  Size: {len(html):,} bytes (self-contained, no external assets)\n")
    print("Done.")


if __name__ == "__main__":
    main()
