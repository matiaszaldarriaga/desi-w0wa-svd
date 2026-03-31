#!/usr/bin/env python3
"""Build the claim provenance dashboard from YAML registries and paper data.

Reads structure/claims.yaml, structure/figures.yaml, structure/scripts.yaml,
data/paper_numbers.json, and paper/sections/*.tex to generate
an interactive HTML dashboard at structure/claim_dashboard.html.

Usage:
    python scripts/build_dashboard.py
"""

import json
import re
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Data loaders ──────────────────────────────────────────────────────────────


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
    with open(PROJECT_ROOT / "data" / "paper_numbers.json") as f:
        return json.load(f)


# ── LaTeX context extraction ─────────────────────────────────────────────────


def extract_claim_contexts():
    """Find all \\claim{id}{...} in .tex files and return context per claim_id."""
    contexts = {}
    sections_dir = PROJECT_ROOT / "paper" / "sections"
    for tex_file in sorted(sections_dir.glob("*.tex")):
        text = tex_file.read_text()
        # Match \claim{id}{text} — text may contain LaTeX commands with braces,
        # so we do a balanced-brace match for the second argument.
        for m in re.finditer(r"\\claim\{([^}]+)\}\{", text):
            claim_id = m.group(1)
            # Find matching closing brace for the second argument
            start_brace = m.end() - 1  # position of the opening {
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

            # Extract surrounding context (~400 chars)
            ctx_start = max(0, m.start() - 200)
            ctx_end = min(len(text), end_brace + 200)
            context_text = text[ctx_start:ctx_end]

            entry = {
                "file": tex_file.name,
                "line": line_num,
                "statement": statement,
                "context": context_text,
            }
            # If claim appears in multiple files, keep all locations
            if claim_id in contexts:
                if isinstance(contexts[claim_id], list):
                    contexts[claim_id].append(entry)
                else:
                    contexts[claim_id] = [contexts[claim_id], entry]
            else:
                contexts[claim_id] = entry
    return contexts


def extract_evidence_annotations():
    """Find all \\evidence{claim_id}{ev_id}{text} in .tex files."""
    evidence = {}
    sections_dir = PROJECT_ROOT / "paper" / "sections"
    for tex_file in sorted(sections_dir.glob("*.tex")):
        text = tex_file.read_text()
        for m in re.finditer(
            r"\\evidence\{([^}]+)\}\{([^}]+)\}\{([^}]*)\}", text
        ):
            claim_id, ev_id, ev_text = m.group(1), m.group(2), m.group(3)
            evidence[ev_id] = {
                "claim_id": claim_id,
                "text": ev_text,
                "file": tex_file.name,
                "line": text[: m.start()].count("\n") + 1,
            }
    return evidence


def extract_dataref_annotations():
    """Find all \\dataref{ev_id}{script_path} in .tex files."""
    datarefs = defaultdict(list)
    sections_dir = PROJECT_ROOT / "paper" / "sections"
    for tex_file in sorted(sections_dir.glob("*.tex")):
        text = tex_file.read_text()
        for m in re.finditer(r"\\dataref\{([^}]+)\}\{([^}]+)\}", text):
            ev_id = m.group(1)
            script_path = m.group(2).replace(r"\_", "_")
            datarefs[ev_id].append(script_path)
    return datarefs


# ── Figure PDF to PNG conversion ─────────────────────────────────────────────


def convert_figures_to_png(figures_yaml):
    """Convert registered figure PDFs to PNGs for browser display.

    Only converts figures listed in figures.yaml — never scans the directory.
    """
    assets_dir = PROJECT_ROOT / "structure" / "dashboard_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PDF paths from the registry
    pdf_paths = []
    for fig in figures_yaml:
        for ff in fig.get("file", "").split(","):
            ff = ff.strip()
            if ff:
                pdf_paths.append(PROJECT_ROOT / ff)

    converted = 0
    skipped = 0
    for pdf in sorted(set(pdf_paths)):
        if not pdf.exists():
            print(f"  Warning: registered figure not found: {pdf}")
            continue
        png = assets_dir / pdf.with_suffix(".png").name
        if not png.exists():
            try:
                subprocess.run(
                    ["sips", "-s", "format", "png", str(pdf), "--out", str(png)],
                    check=True,
                    capture_output=True,
                )
                converted += 1
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"  Warning: could not convert {pdf.name}: {e}")
        else:
            skipped += 1

    total = converted + skipped
    print(f"  Figures: {converted} converted, {skipped} cached, {total} total PNGs")
    return assets_dir


# ── Provenance status checker ────────────────────────────────────────────────


def check_provenance(claim, figures_by_claim, scripts_yaml, evidence_annots, datarefs):
    """Return 'green', 'yellow', or 'red' based on provenance completeness."""
    issues = []
    claim_id = claim["id"]

    # Check evidence exists
    evidence_list = claim.get("evidence", [])
    if not evidence_list:
        issues.append("no evidence items in YAML")

    # Check data_refs point to existing scripts
    for ev in evidence_list:
        refs = ev.get("data_refs", [])
        if not refs:
            issues.append(f"evidence '{ev['id']}' has no data_refs")
        for ref in refs:
            if not (PROJECT_ROOT / ref).exists():
                issues.append(f"missing script: {ref}")

    # Check if claim has associated figures
    if claim_id not in figures_by_claim:
        # Not necessarily bad -- some claims are purely numerical
        pass

    # Check if claim has LaTeX annotation
    if not claim.get("_context"):
        issues.append("no \\claim annotation found in LaTeX")

    if not issues:
        return "green"
    elif len(issues) <= 1:
        return "yellow"
    else:
        return "red"


# ── Number matching ──────────────────────────────────────────────────────────

# Map claim IDs to the paper_numbers.json keys they rely on
CLAIM_NUMBER_KEYS = {
    "c0_universal": {
        "section3_inner_products.min": "0.9969",
        "section3_inner_products.mean": "0.9984",
    },
    "c0_is_omegamh2": {
        "section3_sequential_R2.omh2_only": "0.95",
        "section3_sequential_R2.plus_ombh2": "0.974",
        "section3_sequential_R2.plus_theta": "0.99998",
        "eq6_c0_formula.beta_omh2": "-0.90",
        "eq6_c0_formula.beta_ombh2": "+0.13",
        "eq6_c0_formula.beta_theta": "+0.17",
    },
    "c0_tensions_sign": {
        "table5_c0_tensions.BAO_ACT.tension": "+2.2",
        "table5_c0_tensions.Union3_ACT.tension": "-1.7",
        "table5_c0_tensions.Pantheon+_ACT.tension": "-1.1",
        "table5_c0_tensions.DES-Dovekie_ACT.tension": "-0.8",
    },
    "bao_constrains_omh2": {
        "table3_sigma_c.BAO.sigma_c0": "1.56",
        "table3_sigma_c.Union3.sigma_c0": "0.28",
        "table3_sigma_c.Pantheon+.sigma_c0": "0.30",
        "table3_sigma_c.DES-Dovekie.sigma_c0": "0.33",
    },
    "w0wa_is_c0": {
        "table9_tensions.BAO.c1_tension": "1.2-1.3",
        "table5_c0_tensions.BAO_ACT.tension": "2.2",
    },
    "c0_dominant_w0wa": {
        "table6_grid_ranges.BAO.c0": "104",
        "table6_grid_ranges.BAO.c1": "33.9",
    },
    "freed_calpha_pattern": {},
    "three_mode_ladder": {
        "pivot_fits.c1_BAO.z_pivot": "0.46",
        "pivot_fits.c1_BAO.w_data": "-0.935",
        "pivot_fits.c1_BAO.sigma_w_meas": "0.052",
    },
    "only_omk_measurable": {
        "table13_new_directions.Omk_BAO.sigma_res": "4.27",
    },
    "omk_coherence": {
        "omk_coherence.c0.implied_Omk": "0.0051",
        "omk_coherence.c1.implied_Omk": "0.005",
    },
    "sn_blind_curvature": {
        "table13_new_directions.Omk_Union3.sigma_res": "0.077",
        "table13_new_directions.Omk_Pantheon+.sigma_res": "0.083",
        "table13_new_directions.Omk_DES-Dovekie.sigma_res": "0.072",
    },
    "alens_dilutes": {
        "table12_ext_c0.Alens.tension": "1.1",
        "table12_ext_c0.LCDM.tension": "2.2",
    },
}


def resolve_json_path(numbers, dotted_key):
    """Resolve a dotted key like 'table5_c0_tensions.BAO_ACT.tension' into the JSON value."""
    parts = dotted_key.split(".")
    obj = numbers
    for part in parts:
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def check_number_match(paper_val_str, json_val):
    """Check if a paper value string approximately matches a JSON value."""
    if json_val is None:
        return "missing"
    try:
        # Handle range values like "1.2-1.3"
        if "-" in paper_val_str and paper_val_str.count("-") == 1 and not paper_val_str.startswith("-"):
            lo, hi = paper_val_str.split("-")
            lo_f, hi_f = float(lo), float(hi)
            json_f = float(json_val)
            if lo_f <= json_f <= hi_f:
                return "match"
            else:
                return "mismatch"
        # Handle signed values
        paper_val_str_clean = paper_val_str.lstrip("+")
        paper_f = float(paper_val_str_clean)
        json_f = float(json_val)
        if abs(paper_f - json_f) < 0.051:  # tolerance for rounding
            return "match"
        else:
            return "mismatch"
    except (ValueError, TypeError):
        return "unknown"


def get_claim_numbers(claim_id, numbers):
    """Return list of {key, paper_val, json_val, status} for a claim."""
    mappings = CLAIM_NUMBER_KEYS.get(claim_id, {})
    results = []
    for key, paper_val in mappings.items():
        json_val = resolve_json_path(numbers, key)
        status = check_number_match(paper_val, json_val)
        results.append({
            "key": key,
            "paper_val": paper_val,
            "json_val": json_val,
            "status": status,
        })
    return results


# ── DAG layout (topological) ─────────────────────────────────────────────────


def compute_dag_layout(claims):
    """Compute (x, y) positions for DAG nodes using layered layout."""
    claim_ids = [c["id"] for c in claims]
    depends = {}
    for c in claims:
        depends[c["id"]] = c.get("depends_on", [])

    # Assign layers by longest path from root
    layers = {}

    def get_layer(cid, visited=None):
        if visited is None:
            visited = set()
        if cid in layers:
            return layers[cid]
        if cid in visited:
            return 0  # cycle guard
        visited.add(cid)
        deps = depends.get(cid, [])
        if not deps:
            layers[cid] = 0
            return 0
        max_dep = max(get_layer(d, visited) for d in deps if d in depends)
        layers[cid] = max_dep + 1
        return layers[cid]

    for cid in claim_ids:
        get_layer(cid)

    # Group by layer
    layer_groups = defaultdict(list)
    for cid, layer in layers.items():
        layer_groups[layer].append(cid)

    # Assign positions
    node_w = 160
    node_h = 60
    x_gap = 30
    y_gap = 100
    positions = {}

    max_nodes_in_layer = max(len(v) for v in layer_groups.values())
    total_width = max_nodes_in_layer * (node_w + x_gap)

    for layer_idx in sorted(layer_groups.keys()):
        nodes = layer_groups[layer_idx]
        n = len(nodes)
        layer_width = n * node_w + (n - 1) * x_gap
        start_x = (total_width - layer_width) / 2
        for i, cid in enumerate(nodes):
            x = start_x + i * (node_w + x_gap)
            y = layer_idx * (node_h + y_gap)
            positions[cid] = (x, y)

    return positions, node_w, node_h, layers


def generate_dag_svg(claims, colors):
    """Generate an SVG DAG from claim dependencies."""
    positions, node_w, node_h, layers = compute_dag_layout(claims)
    depends = {c["id"]: c.get("depends_on", []) for c in claims}

    # Short labels for nodes
    short_labels = {
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
    }

    # Section labels
    section_map = {}
    for c in claims:
        sec = c.get("section", "")
        # Extract just the section number
        m = re.match(r"([\S]+\d+)", sec)
        section_map[c["id"]] = m.group(1) if m else sec.split()[0] if sec else ""

    color_map = {"green": "#2e7d32", "yellow": "#f9a825", "red": "#c62828"}
    bg_map = {"green": "#e8f5e9", "yellow": "#fff8e1", "red": "#ffebee"}
    border_map = {"green": "#81c784", "yellow": "#fdd835", "red": "#ef5350"}

    # Compute SVG dimensions
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    svg_w = max(all_x) + node_w + 40 if all_x else 800
    svg_h = max(all_y) + node_h + 40 if all_y else 400

    lines = []
    lines.append(
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'width="100%" style="max-width:{int(svg_w)}px; display:block; margin:0 auto;">'
    )
    lines.append("  <defs>")
    lines.append(
        '    <marker id="arrowhead" markerWidth="10" markerHeight="7" '
        'refX="10" refY="3.5" orient="auto">'
    )
    lines.append('      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>')
    lines.append("    </marker>")
    lines.append("  </defs>")

    # Draw edges (from dependency to dependent)
    for cid, deps in depends.items():
        if cid not in positions:
            continue
        cx, cy = positions[cid]
        for dep in deps:
            if dep not in positions:
                continue
            dx, dy = positions[dep]
            # Arrow from dep (parent) to cid (child)
            x1 = dx + node_w / 2
            y1 = dy + node_h
            x2 = cx + node_w / 2
            y2 = cy
            # Simple bezier for curved edges
            mid_y = (y1 + y2) / 2
            lines.append(
                f'  <path d="M {x1},{y1} C {x1},{mid_y} {x2},{mid_y} {x2},{y2}" '
                f'fill="none" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>'
            )

    # Draw nodes
    for c in claims:
        cid = c["id"]
        if cid not in positions:
            continue
        x, y = positions[cid]
        clr = colors.get(cid, "yellow")
        bg = bg_map[clr]
        border = border_map[clr]
        label = short_labels.get(cid, cid)
        sec = section_map.get(cid, "")

        lines.append(
            f'  <g class="dag-node" onclick="scrollToPanel(\'{cid}\')" '
            f'style="cursor:pointer">'
        )
        lines.append(
            f'    <rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" '
            f'rx="8" ry="8" fill="{bg}" stroke="{border}" stroke-width="2"/>'
        )
        # Label text
        lines.append(
            f'    <text x="{x + node_w/2}" y="{y + 22}" text-anchor="middle" '
            f'font-size="11" font-weight="600" fill="#333">{_esc(label)}</text>'
        )
        # Section text
        lines.append(
            f'    <text x="{x + node_w/2}" y="{y + 40}" text-anchor="middle" '
            f'font-size="10" fill="#666">{_esc(sec)}</text>'
        )
        # Status dot
        dot_clr = color_map[clr]
        lines.append(
            f'    <circle cx="{x + node_w - 12}" cy="{y + 12}" r="5" fill="{dot_clr}"/>'
        )
        lines.append("  </g>")

    lines.append("</svg>")
    return "\n".join(lines)


def _esc(s):
    """Escape HTML special characters."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── HTML generation ──────────────────────────────────────────────────────────


def _clean_latex(s):
    """Strip common LaTeX commands for display in HTML."""
    s = s.replace("\\czero", "c0")
    s = s.replace("\\cone", "c1")
    s = s.replace("\\omh", "Omega_m h^2")
    s = s.replace("\\obh", "Omega_b h^2")
    s = s.replace("\\thetastar", "theta*")
    s = s.replace("\\Omk", "Omega_k")
    s = s.replace("\\wowa", "w0wa")
    s = s.replace("\\LCDM", "LCDM")
    s = s.replace("\\sigres", "sigma_res")
    s = s.replace("\\mathrm{lens}", "lens")
    s = s.replace("\\sim", "~")
    s = s.replace("\\pm", "+/-")
    s = s.replace("\\sigma", "sigma")
    s = s.replace("\\approx", "~")
    s = re.sub(r"\$([^$]*)\$", r"\1", s)  # strip $...$
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)  # strip \cmd{arg} -> arg
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # strip remaining \commands
    s = re.sub(r"\{|\}", "", s)  # strip remaining braces
    return s.strip()


def generate_detail_panel(claim, figures_by_claim, scripts_by_path, numbers, colors,
                          evidence_annots, datarefs):
    """Generate an HTML detail panel for a single claim."""
    cid = claim["id"]
    clr = colors.get(cid, "yellow")
    statement = claim.get("statement", "")
    section = claim.get("section", "")

    # Location info
    ctx = claim.get("_context")
    if isinstance(ctx, list):
        loc_parts = [f"{c['file']}:{c['line']}" for c in ctx]
        location = ", ".join(loc_parts)
    elif ctx:
        location = f"{ctx['file']}:{ctx['line']}"
    else:
        location = claim.get("file", "unknown")

    lines = []
    lines.append(f'<div id="panel-{cid}" class="claim-panel {clr}">')
    lines.append(f'  <div class="panel-header" onclick="togglePanel(\'{cid}\')">')
    lines.append(f'    <span class="status-dot {clr}"></span>')
    lines.append(f"    <strong>{_esc(cid)}</strong>")
    lines.append(f'    <span class="section-tag">{_esc(section)}</span>')
    lines.append(f'    <span class="expand-icon" id="icon-{cid}">+</span>')
    lines.append("  </div>")
    lines.append(f'  <p class="claim-statement">{_esc(_clean_latex(statement))}</p>')

    lines.append(f'  <div id="detail-{cid}" class="panel-detail" style="display:none">')

    # Location
    lines.append(f"    <h4>Location</h4>")
    if isinstance(ctx, list):
        for c in ctx:
            rel = f"../paper/sections/{c['file']}"
            lines.append(f'    <p><a href="{rel}">{_esc(c["file"])}:{c["line"]}</a></p>')
    elif ctx:
        rel = f"../paper/sections/{ctx['file']}"
        lines.append(f'    <p><a href="{rel}">{_esc(ctx["file"])}:{ctx["line"]}</a></p>')

    # Evidence
    evidence_list = claim.get("evidence", [])
    if evidence_list:
        lines.append("    <h4>Evidence</h4>")
        lines.append("    <ul>")
        for ev in evidence_list:
            ev_id = ev.get("id", "")
            ev_desc = ev.get("description", "")
            refs = ev.get("data_refs", [])
            lines.append(f"      <li>")
            lines.append(f"        <strong>{_esc(ev_id)}</strong>: {_esc(ev_desc)}")
            if refs:
                script_links = []
                for ref in refs:
                    rel_path = f"../{ref}"
                    name = Path(ref).name
                    exists = (PROJECT_ROOT / ref).exists()
                    if exists:
                        script_links.append(f'<a href="{rel_path}">{_esc(name)}</a>')
                    else:
                        script_links.append(
                            f'<span class="missing">{_esc(name)} (missing)</span>'
                        )
                lines.append(f"        <br>Scripts: {', '.join(script_links)}")
            lines.append("      </li>")
        lines.append("    </ul>")

    # Figures
    figs = figures_by_claim.get(cid, [])
    if figs:
        lines.append("    <h4>Figures</h4>")
        lines.append('    <div class="figures-grid">')
        for fig in figs:
            fig_file = fig.get("file", "")
            # Handle comma-separated file lists
            fig_files = [f.strip() for f in fig_file.split(",")]
            for ff in fig_files:
                pdf_name = Path(ff).name
                png_name = Path(ff).stem + ".png"
                label = fig.get("label", "")
                lines.append(f'      <div class="fig-card">')
                lines.append(
                    f'        <img src="dashboard_assets/{png_name}" '
                    f'alt="{_esc(label)}" loading="lazy">'
                )
                lines.append(f"        <p>{_esc(label)}</p>")
                lines.append(f"      </div>")
        lines.append("    </div>")

    # Numbers
    num_entries = get_claim_numbers(cid, numbers)
    if num_entries:
        lines.append("    <h4>Numbers</h4>")
        lines.append('    <table class="numbers-table">')
        lines.append(
            "      <tr><th>JSON Key</th><th>Paper Value</th>"
            "<th>JSON Value</th><th>Status</th></tr>"
        )
        for entry in num_entries:
            status = entry["status"]
            status_class = {
                "match": "num-match",
                "mismatch": "num-mismatch",
                "missing": "num-missing",
                "unknown": "num-unknown",
            }.get(status, "")
            status_icon = {
                "match": "OK",
                "mismatch": "MISMATCH",
                "missing": "N/A",
                "unknown": "?",
            }.get(status, "?")
            json_display = (
                str(entry["json_val"]) if entry["json_val"] is not None else "---"
            )
            lines.append(
                f'      <tr class="{status_class}">'
                f"<td><code>{_esc(entry['key'])}</code></td>"
                f"<td>{_esc(entry['paper_val'])}</td>"
                f"<td>{_esc(json_display)}</td>"
                f"<td>{status_icon}</td></tr>"
            )
        lines.append("    </table>")

    # Dependencies
    deps = claim.get("depends_on", [])
    if deps:
        lines.append("    <h4>Dependencies</h4>")
        dep_links = [
            f'<a href="#panel-{d}" onclick="scrollToPanel(\'{d}\')">{_esc(d)}</a>'
            for d in deps
        ]
        lines.append(f"    <p>Depends on: {', '.join(dep_links)}</p>")

    # Depended on by
    dependents = claim.get("_dependents", [])
    if dependents:
        dep_links = [
            f'<a href="#panel-{d}" onclick="scrollToPanel(\'{d}\')">{_esc(d)}</a>'
            for d in dependents
        ]
        lines.append(f"    <p>Required by: {', '.join(dep_links)}</p>")

    lines.append("  </div>")  # panel-detail
    lines.append("</div>")  # claim-panel

    return "\n".join(lines)


def generate_summary_bar(claims, colors, figures_yaml, claims_yaml):
    """Generate summary statistics bar."""
    n_claims = len(claims)
    n_green = sum(1 for c in claims if colors.get(c["id"]) == "green")
    n_yellow = sum(1 for c in claims if colors.get(c["id"]) == "yellow")
    n_red = sum(1 for c in claims if colors.get(c["id"]) == "red")

    n_figures = len(figures_yaml)

    return f"""
    <div class="summary-bar">
      <div class="stat">
        <span class="stat-number">{n_claims}</span>
        <span class="stat-label">Claims</span>
      </div>
      <div class="stat">
        <span class="stat-number green-text">{n_green}</span>
        <span class="stat-label">Complete</span>
      </div>
      <div class="stat">
        <span class="stat-number yellow-text">{n_yellow}</span>
        <span class="stat-label">Partial</span>
      </div>
      <div class="stat">
        <span class="stat-number red-text">{n_red}</span>
        <span class="stat-label">Missing</span>
      </div>
      <div class="stat-divider"></div>
      <div class="stat">
        <span class="stat-number">{n_figures}</span>
        <span class="stat-label">Registered Figures</span>
      </div>
    </div>
    """


def generate_unlinked_list(figures_yaml):
    """Generate list of registered figures not linked to any claim."""
    unlinked = [
        fig for fig in figures_yaml
        if not fig.get("supports_claims")
    ]

    if not unlinked:
        return ""

    lines = ['<div class="orphan-section">']
    lines.append("  <h3>Registered Figures Not Linked to Any Claim</h3>")
    lines.append("  <p>These figures are in the paper (figures.yaml) but have no <code>supports_claims</code> entry. Consider whether they should be linked to a claim or whether they are purely illustrative.</p>")
    lines.append('  <div class="figures-grid">')
    for fig in unlinked:
        label = fig.get("label", "unknown")
        section = fig.get("section", "?")
        for ff in fig.get("file", "").split(","):
            ff = ff.strip()
            if not ff:
                continue
            pdf_name = Path(ff).name
            png_name = Path(pdf_name).stem + ".png"
            lines.append(f'    <div class="fig-card orphan">')
            lines.append(
                f'      <img src="dashboard_assets/{png_name}" '
                f'alt="{_esc(pdf_name)}" loading="lazy">'
            )
            lines.append(f"      <p>{_esc(label)} ({section})</p>")
            lines.append(f"    </div>")
    lines.append("  </div>")
    lines.append("</div>")
    return "\n".join(lines)


CSS = """
:root {
  --green: #2e7d32;
  --green-bg: #e8f5e9;
  --green-border: #81c784;
  --yellow: #f9a825;
  --yellow-bg: #fff8e1;
  --yellow-border: #fdd835;
  --red: #c62828;
  --red-bg: #ffebee;
  --red-border: #ef5350;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  max-width: 1200px; margin: 0 auto; padding: 20px;
  background: #fafafa; color: #333;
  line-height: 1.5;
}
h1 { margin-bottom: 4px; }
.subtitle { color: #666; margin-top: 0; margin-bottom: 24px; }

/* Summary bar */
.summary-bar {
  display: flex; align-items: center; gap: 24px;
  background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
  padding: 16px 24px; margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.stat { text-align: center; }
.stat-number { display: block; font-size: 28px; font-weight: 700; }
.stat-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-divider { width: 1px; height: 40px; background: #e0e0e0; }
.green-text { color: var(--green); }
.yellow-text { color: var(--yellow); }
.red-text { color: var(--red); }

/* DAG section */
.dag-section {
  background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
  padding: 20px; margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  overflow-x: auto;
}
.dag-section h2 { margin-top: 0; }

/* Claim panels */
.claim-panel {
  border: 1px solid #e0e0e0; border-radius: 8px;
  margin: 8px 0; background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  transition: box-shadow 0.2s;
}
.claim-panel:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
.claim-panel.green { border-left: 4px solid var(--green-border); }
.claim-panel.yellow { border-left: 4px solid var(--yellow-border); }
.claim-panel.red { border-left: 4px solid var(--red-border); }

.panel-header {
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px; cursor: pointer; user-select: none;
}
.panel-header:hover { background: #f5f5f5; border-radius: 8px; }
.status-dot {
  width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
}
.status-dot.green { background: var(--green); }
.status-dot.yellow { background: var(--yellow); }
.status-dot.red { background: var(--red); }
.section-tag {
  margin-left: auto; font-size: 12px; color: #888;
  background: #f0f0f0; padding: 2px 8px; border-radius: 4px;
}
.expand-icon {
  font-size: 18px; font-weight: 300; color: #999;
  width: 24px; text-align: center;
}

.claim-statement {
  padding: 0 16px 8px 36px; margin: 0;
  font-size: 14px; color: #555;
}
.panel-detail {
  padding: 0 16px 16px 36px;
  border-top: 1px solid #eee;
}
.panel-detail h4 {
  margin: 16px 0 8px 0; font-size: 13px; text-transform: uppercase;
  letter-spacing: 0.5px; color: #888;
}

/* Figures grid */
.figures-grid {
  display: flex; flex-wrap: wrap; gap: 12px;
}
.fig-card {
  border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px;
  background: #fafafa; max-width: 380px;
}
.fig-card img { max-width: 100%; height: auto; border-radius: 4px; }
.fig-card p { margin: 6px 0 0 0; font-size: 12px; color: #666; text-align: center; }
.fig-card.orphan { border-color: #ef5350; background: #fff5f5; }

/* Numbers table */
.numbers-table {
  border-collapse: collapse; width: 100%; font-size: 13px;
}
.numbers-table th {
  background: #f5f5f5; padding: 6px 10px; text-align: left;
  border-bottom: 2px solid #ddd; font-weight: 600;
}
.numbers-table td { padding: 5px 10px; border-bottom: 1px solid #eee; }
.numbers-table code { font-size: 12px; background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
.num-match td:last-child { color: var(--green); font-weight: 600; }
.num-mismatch { background: var(--red-bg); }
.num-mismatch td:last-child { color: var(--red); font-weight: 600; }
.num-missing td:last-child { color: #999; }

/* Links and misc */
a { color: #1565c0; text-decoration: none; }
a:hover { text-decoration: underline; }
.missing { color: var(--red); font-style: italic; }
ul { padding-left: 20px; }
li { margin-bottom: 6px; }

/* Orphan section */
.orphan-section {
  background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
  padding: 20px; margin-top: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.orphan-section h3 { margin-top: 0; color: var(--red); }

/* Highlight animation */
.highlight { animation: flash 1s ease-out; }
@keyframes flash {
  0% { background: #fff9c4; }
  100% { background: transparent; }
}

/* Footer */
.footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #eee;
  font-size: 12px; color: #999; text-align: center; }
"""

JS = """
function togglePanel(id) {
  var detail = document.getElementById('detail-' + id);
  var icon = document.getElementById('icon-' + id);
  if (detail.style.display === 'none') {
    detail.style.display = 'block';
    icon.textContent = String.fromCharCode(8722); // minus sign
  } else {
    detail.style.display = 'none';
    icon.textContent = '+';
  }
}

function scrollToPanel(id) {
  var panel = document.getElementById('panel-' + id);
  if (!panel) return;
  // Expand it
  var detail = document.getElementById('detail-' + id);
  var icon = document.getElementById('icon-' + id);
  if (detail && detail.style.display === 'none') {
    detail.style.display = 'block';
    if (icon) icon.textContent = String.fromCharCode(8722);
  }
  // Scroll
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  // Flash highlight
  panel.classList.add('highlight');
  setTimeout(function() { panel.classList.remove('highlight'); }, 1200);
}

// Expand all / collapse all
function expandAll() {
  document.querySelectorAll('.panel-detail').forEach(function(el) {
    el.style.display = 'block';
  });
  document.querySelectorAll('.expand-icon').forEach(function(el) {
    el.textContent = String.fromCharCode(8722);
  });
}
function collapseAll() {
  document.querySelectorAll('.panel-detail').forEach(function(el) {
    el.style.display = 'none';
  });
  document.querySelectorAll('.expand-icon').forEach(function(el) {
    el.textContent = '+';
  });
}
"""


def generate_html(claims, figures_yaml, scripts_yaml, numbers, colors,
                  evidence_annots, datarefs):
    """Generate the complete dashboard HTML."""
    # Build figures-by-claim lookup
    figures_by_claim = defaultdict(list)
    for fig in figures_yaml:
        for cid in fig.get("supports_claims", []):
            figures_by_claim[cid].append(fig)

    # Build scripts-by-path lookup
    scripts_by_path = {}
    for s in scripts_yaml:
        scripts_by_path[s.get("path", "")] = s

    # Compute dependents (reverse of depends_on)
    for c in claims:
        c["_dependents"] = []
    claim_map = {c["id"]: c for c in claims}
    for c in claims:
        for dep in c.get("depends_on", []):
            if dep in claim_map:
                claim_map[dep]["_dependents"].append(c["id"])

    dag_svg = generate_dag_svg(claims, colors)
    summary = generate_summary_bar(claims, colors, figures_yaml, claims)
    orphans = generate_unlinked_list(figures_yaml)

    # Generate panels grouped by section
    section_order = [
        "1", "intro",
        "2", "data",
        "3", "universal",
        "4", "tension",
        "5", "w0wa", "reinterpretation",
        "6", "curvature",
        "7", "extension",
        "8", "conclusion",
        "App", "appendix",
    ]

    def section_sort_key(claim):
        sec = claim.get("section", "").lower()
        for i, kw in enumerate(section_order):
            if kw.lower() in sec:
                return i
        return 999

    sorted_claims = sorted(claims, key=section_sort_key)

    panels = []
    for c in sorted_claims:
        panel = generate_detail_panel(
            c, figures_by_claim, scripts_by_path, numbers, colors,
            evidence_annots, datarefs
        )
        panels.append(panel)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
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
<p class="subtitle">DESI-w0wa paper -- generated {timestamp}</p>

{summary}

<div class="dag-section">
  <h2>Claim Dependency Graph</h2>
  <p style="font-size:13px; color:#888; margin-top:-8px;">
    Click any node to scroll to its detail panel.
    Green = fully traced, Yellow = partial, Red = missing provenance.
  </p>
  {dag_svg}
</div>

<div style="display:flex; gap:8px; margin-bottom:12px;">
  <button onclick="expandAll()" style="padding:6px 14px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:13px;">Expand All</button>
  <button onclick="collapseAll()" style="padding:6px 14px; border:1px solid #ccc; border-radius:4px; background:#fff; cursor:pointer; font-size:13px;">Collapse All</button>
</div>

<h2>Claim Details</h2>

{"".join(panels)}

{orphans}

<div class="footer">
  Generated by <code>scripts/build_dashboard.py</code> from YAML registries and paper data.
</div>

<script>
{JS}
</script>

</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print("Building claim provenance dashboard...")
    print()

    # Load data
    print("Loading registries...")
    claims = load_claims()
    figures = load_figures()
    scripts = load_scripts()
    numbers = load_numbers()
    print(f"  Claims: {len(claims)}")
    print(f"  Figures: {len(figures)}")
    print(f"  Scripts: {len(scripts)}")
    print(f"  Number keys: {len(numbers)}")
    print()

    # Extract LaTeX annotations
    print("Extracting LaTeX annotations...")
    contexts = extract_claim_contexts()
    evidence_annots = extract_evidence_annotations()
    datarefs = extract_dataref_annotations()
    print(f"  Claim annotations found: {len(contexts)}")
    print(f"  Evidence annotations found: {len(evidence_annots)}")
    print(f"  Dataref annotations found: {sum(len(v) for v in datarefs.values())}")
    print()

    # Enrich claims with context
    for claim in claims:
        cid = claim["id"]
        claim["_context"] = contexts.get(cid)

    # Convert figures
    print("Converting figures to PNG...")
    assets_dir = convert_figures_to_png(figures)
    print()

    # Compute provenance colors
    print("Checking provenance status...")
    figures_by_claim = defaultdict(list)
    for fig in figures:
        for cid in fig.get("supports_claims", []):
            figures_by_claim[cid].append(fig)

    colors = {}
    for claim in claims:
        clr = check_provenance(
            claim, figures_by_claim, scripts, evidence_annots, datarefs
        )
        colors[claim["id"]] = clr
        print(f"  {claim['id']}: {clr}")
    print()

    # Generate HTML
    print("Generating HTML dashboard...")
    html = generate_html(
        claims, figures, scripts, numbers, colors, evidence_annots, datarefs
    )
    out_path = PROJECT_ROOT / "structure" / "claim_dashboard.html"
    out_path.write_text(html)
    print(f"  Written to: {out_path}")
    print(f"  Size: {len(html):,} bytes")
    print()

    # Summary
    n_green = sum(1 for v in colors.values() if v == "green")
    n_yellow = sum(1 for v in colors.values() if v == "yellow")
    n_red = sum(1 for v in colors.values() if v == "red")
    n_pngs = len(list(assets_dir.glob("*.png")))
    print("Summary:")
    print(f"  {len(claims)} claims: {n_green} green, {n_yellow} yellow, {n_red} red")
    print(f"  {n_pngs} PNG figure assets in {assets_dir}")
    print(f"  Dashboard: {out_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
