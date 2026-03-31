#!/bin/bash
# Download all external data needed to reproduce the paper.
# Run from the repository root: bash scripts/download_data.sh
#
# Data sources:
#   - DESI DR2 BAO:       CobayaSampler/bao_data (GitHub)
#   - SN compilations:    CobayaSampler/sn_data (GitHub)
#   - ACT DR6 chains:     NERSC portal
#   - Planck 2018 chains: Planck Legacy Archive (ESA)
#   - DESI DR2 w0wa:      DESI public data release
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data"

echo "========================================"
echo " Downloading data to $DATA_DIR"
echo "========================================"
echo ""

# ── BAO data (DESI DR2) ──────────────────────────────────────
if [ ! -d "$DATA_DIR/bao_data" ]; then
    echo "[1/5] Cloning BAO data (CobayaSampler/bao_data)..."
    git clone --depth 1 https://github.com/CobayaSampler/bao_data "$DATA_DIR/bao_data"
else
    echo "[1/5] BAO data already present, skipping."
fi

# ── SN data (Pantheon+, DES Y5, Union3) ─────────────────────
if [ ! -d "$DATA_DIR/sn_data" ]; then
    echo "[2/5] Cloning SN data (CobayaSampler/sn_data)..."
    git clone --depth 1 https://github.com/CobayaSampler/sn_data "$DATA_DIR/sn_data"
else
    echo "[2/5] SN data already present, skipping."
fi

# ── ACT DR6 chains ───────────────────────────────────────────
echo "[3/5] ACT DR6 chains..."
ACT_BASE="https://portal.nersc.gov/project/act/dr6.02/chains"
ACT_CHAINS=(
    p-actbase_lcdm_camb
    p-actbase_ok_camb
    p-actbase_alens_camb
    p-actbase_bprim_class
    p-actbase_ede_n2_camb
)
for chain in "${ACT_CHAINS[@]}"; do
    DEST="$DATA_DIR/chains/$chain"
    if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ]; then
        echo "  Downloading $chain..."
        mkdir -p "$DEST"
        wget -q -r -np -nH --cut-dirs=5 -P "$DEST" \
             --reject "index.html*" "$ACT_BASE/$chain/"
    else
        echo "  $chain already present, skipping."
    fi
done

# ── Planck 2018 chains ───────────────────────────────────────
PLANCK_DEST="$DATA_DIR/chains/COM_CosmoParams_fullGrid_R3.01"
if [ ! -d "$PLANCK_DEST" ]; then
    echo "[4/5] Downloading Planck 2018 chains (COM_CosmoParams_fullGrid_R3.01)..."
    mkdir -p "$PLANCK_DEST"
    PLANCK_URL="https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_fullGrid_R3.01.tar.gz"
    TMP_FILE="$(mktemp /tmp/planck_chains_XXXXXX.tar.gz)"
    wget -q -O "$TMP_FILE" "$PLANCK_URL"
    tar xzf "$TMP_FILE" -C "$DATA_DIR/chains/"
    rm -f "$TMP_FILE"
else
    echo "[4/5] Planck chains already present, skipping."
fi

# ── DESI DR2 w0wa chain ─────────────────────────────────────
DESI_DEST="$DATA_DIR/chains/desi_dr2_official"
if [ ! -d "$DESI_DEST" ]; then
    echo "[5/5] Downloading DESI DR2 chains (w0wa + omegak)..."
    DESI_BASE="https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params"
    # w0wa chain
    mkdir -p "$DESI_DEST/base_w_wa"
    wget -q -r -np -nH --cut-dirs=6 -P "$DESI_DEST/base_w_wa" \
         --reject "index.html*" "$DESI_BASE/chains/base_w_wa/"
    # omegak chain
    mkdir -p "$DESI_DEST/base_omegak"
    wget -q -r -np -nH --cut-dirs=6 -P "$DESI_DEST/base_omegak" \
         --reject "index.html*" "$DESI_BASE/chains/base_omegak/"
else
    echo "[5/5] DESI chains already present, skipping."
fi

echo ""
echo "========================================"
echo " Download complete"
echo "========================================"
echo "BAO:    $DATA_DIR/bao_data/"
echo "SN:     $DATA_DIR/sn_data/"
echo "Chains: $DATA_DIR/chains/"
echo ""
echo "Total chain directories:"
ls -1d "$DATA_DIR/chains"/*/ 2>/dev/null | wc -l | tr -d ' '
