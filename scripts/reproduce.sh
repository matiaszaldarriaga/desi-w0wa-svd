#!/bin/bash
# Reproduce all paper figures and numbers from raw data.
#
# Prerequisites:
#   pip install -e .
#   bash scripts/download_data.sh
#
# This script runs the three-step pipeline:
#   1. precompute_figure_data.py  — chains -> data/figure_data.npz  (~2 min)
#   2. compute_paper_numbers.py   — chains -> data/paper_numbers.json (~2 min)
#   3. make_all_figures.py        — precomputed data -> 18 PDFs     (< 5 sec)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " Reproducing paper figures and numbers"
echo "========================================"
echo ""

# ── Step 1: Precompute figure data ──
echo "=== Step 1/3: Precompute figure data ==="
echo "  (chains -> data/figure_data.npz, ~2 min)"
echo ""
python "$SCRIPT_DIR/precompute_figure_data.py"
echo ""

# ── Step 2: Compute paper numbers ──
echo "=== Step 2/3: Compute paper numbers ==="
echo "  (chains -> data/paper_numbers.json, ~2 min)"
echo ""
python "$SCRIPT_DIR/compute_paper_numbers.py"
echo ""

# ── Step 3: Generate all figures ──
echo "=== Step 3/3: Generate all figures ==="
echo "  (precomputed data -> paper/figures/*.pdf, < 5 sec)"
echo ""
python "$SCRIPT_DIR/make_all_figures.py"
echo ""

# ── Verification ──
echo "========================================"
echo " Verification"
echo "========================================"

FIGS="$ROOT_DIR/paper/figures"
EXPECTED=18

# Count PDFs
N_PDF=$(find "$FIGS" -name "*.pdf" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "Figures produced: $N_PDF / $EXPECTED"

# List all figures
if [ "$N_PDF" -gt 0 ]; then
    echo ""
    echo "Generated figures:"
    ls -1 "$FIGS"/*.pdf 2>/dev/null | while read -r f; do
        echo "  $(basename "$f")"
    done
fi

echo ""
if [ "$N_PDF" -eq "$EXPECTED" ]; then
    echo "SUCCESS: All $EXPECTED figures generated."
else
    echo "WARNING: Expected $EXPECTED figures, found $N_PDF."
    echo ""
    echo "Missing figures:"
    for fig in \
        fig_dv_ratios.pdf fig_dmdh_ratios.pdf fig_sn_datasets.pdf \
        fig_c0_families.pdf fig_c0_gaussians.pdf fig_omegamh2_money.pdf \
        fig_hz_omh2_variation.pdf fig_w0wa_contours.pdf \
        sec5_sn_all_datasets.pdf sec5_sn_c1_histogram.pdf \
        sec5_bao_data_bands_dv.pdf sec5_bao_data_bands_dmdh.pdf \
        fig_w0wa_chi2_investigation.pdf fig_wp_histograms.pdf \
        fig_omk_gaussians.pdf fig_ext_c0_dists.pdf \
        fig_app_derivatives.pdf fig_app_beta_prediction.pdf; do
        if [ ! -f "$FIGS/$fig" ]; then
            echo "  MISSING: $fig"
        fi
    done
    exit 1
fi

echo ""
echo "Output locations:"
echo "  Paper numbers:  $ROOT_DIR/data/paper_numbers.json"
echo "  Figure data:    $ROOT_DIR/data/figure_data.npz"
echo "  Figures:        $FIGS/"
echo ""
echo "To compile the paper:"
echo "  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
