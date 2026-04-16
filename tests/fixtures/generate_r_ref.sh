#!/usr/bin/env bash
# Generate R-STILT reference fixtures for the footprint fidelity test.
#
# Runs the official uataq/stilt with the 01-wbb tutorial met data and WBB
# receptor, then copies the trajectory parquet and footprint NetCDF into
# tests/fixtures/r_ref/ so they can be committed and used by
# tests/test_footprint_fidelity.py without needing R or met files on CI.
#
# Usage (from the PYSTILT repo root):
#   bash tests/fixtures/generate_r_ref.sh
#
# Optional environment variables:
#   STILT_R_DIR    Path to uataq/stilt checkout (default: ../R-STILT)
#   STILT_MET_DIR  Path to stilt-tutorials/01-wbb/met dir
#                  (default: ../stilt-tutorials/01-wbb/met)
#
# R package dependencies (installed automatically if missing):
#   devtools, uataq (from github.com/uataq/uataq), arrow (for RDS→parquet conversion)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/r_ref"

STILT_R_DIR="${STILT_R_DIR:-$REPO_ROOT/../R-STILT}"
STILT_TUTORIALS_DIR="${STILT_TUTORIALS_DIR:-$REPO_ROOT/../stilt-tutorials}"
STILT_MET_DIR="${STILT_MET_DIR:-$STILT_TUTORIALS_DIR/01-wbb/met}"

SIM_ID="201512100000_-112_40.5_5"
OUT_DIR="$STILT_R_DIR/out/by-id/$SIM_ID"
RUN_STILT="$STILT_R_DIR/r/run_stilt.r"

echo "=== Generating R-STILT reference fixtures ==="
echo "R-STILT dir : $STILT_R_DIR"
echo "Met dir     : $STILT_MET_DIR"
echo "Fixtures dir: $FIXTURES_DIR"
echo ""

# Install/ensure R package dependencies
echo "Ensuring R package dependencies (devtools, uataq, arrow) ..."
Rscript -e "
  if (!require('devtools', quietly=TRUE)) install.packages('devtools', repos='https://cloud.r-project.org/')
  if (!require('uataq', quietly=TRUE)) devtools::install_github('uataq/uataq')
  if (!require('arrow', quietly=TRUE)) install.packages('arrow', repos='https://cloud.r-project.org/')
"

# Clone stilt-tutorials if the met dir is missing
if [[ ! -d "$STILT_MET_DIR" ]]; then
  echo "Met dir not found. Cloning uataq/stilt-tutorials ..."
  git clone https://github.com/uataq/stilt-tutorials "$STILT_TUTORIALS_DIR"
fi

# Initialize R-STILT if not already present
if [[ ! -d "$STILT_R_DIR" ]]; then
  echo "R-STILT directory not found. Initializing via uataq::stilt_init ..."
  PARENT_DIR="$(dirname "$STILT_R_DIR")"
  STILT_NAME="$(basename "$STILT_R_DIR")"
  Rscript -e "setwd('$PARENT_DIR'); require('uataq'); uataq::stilt_init('$STILT_NAME')"
fi

# Patch run_stilt.r (creates .bak for restore on exit)
cp "$RUN_STILT" "$RUN_STILT.bak"

cleanup() {
  if [[ -f "$RUN_STILT.bak" ]]; then
    mv "$RUN_STILT.bak" "$RUN_STILT"
    echo "Restored run_stilt.r"
  fi
}
trap cleanup EXIT

sed -i \
  -e "s|met_path           <- '<path_to_arl_meteorological_data>'|met_path           <- '$STILT_MET_DIR'|" \
  -e "s|n_hours       <- -24|n_hours       <- -6|" \
  -e "s|xmn <- NA|xmn <- -113|" \
  -e "s|xmx <- NA|xmx <- -111|" \
  -e "s|ymn <- NA|ymn <- 39.5|" \
  -e "s|ymx <- NA|ymx <- 41.5|" \
  "$RUN_STILT"

# Run R-STILT from its working directory
echo "Running Rscript r/run_stilt.r in $STILT_R_DIR ..."
pushd "$STILT_R_DIR" > /dev/null
Rscript r/run_stilt.r
popd > /dev/null

# Verify outputs exist
TRAJ_RDS="$OUT_DIR/${SIM_ID}_traj.rds"
FOOT_NC=$(ls "$OUT_DIR/${SIM_ID}_foot.nc" 2>/dev/null | head -1)

if [[ ! -f "$TRAJ_RDS" ]]; then
  echo "ERROR: _traj.rds not found. Check $OUT_DIR"
  exit 1
fi
if [[ -z "$FOOT_NC" ]]; then
  echo "ERROR: _foot.nc not found. Check $OUT_DIR"
  exit 1
fi

# Convert RDS trajectory to parquet so Python can read it
TRAJ_PARQUET="$OUT_DIR/r_traj.parquet"
echo "Converting traj.rds -> traj.parquet ..."
Rscript -e "
  library(arrow)
  obj <- readRDS('$TRAJ_RDS')
  traj <- if (is.data.frame(obj)) obj else obj\$particle
  write_parquet(traj, '$TRAJ_PARQUET')
  cat('Written:', nrow(traj), 'rows to $TRAJ_PARQUET\n')
"

# Copy to fixtures (overwrite any previous reference)
mkdir -p "$FIXTURES_DIR"
cp "$TRAJ_PARQUET" "$FIXTURES_DIR/${SIM_ID}_traj.parquet"
cp "$FOOT_NC"      "$FIXTURES_DIR/r_foot.nc"

echo ""
echo "=== Fixtures written ==="
ls -lh "$FIXTURES_DIR"
echo ""
echo "Commit these files to lock the reference:"
echo "  git add tests/fixtures/r_ref/"
echo "  git commit -m 'test: update R-STILT reference fixtures'"
