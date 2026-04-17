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
#   STILT_KRAND    HYSPLIT KRAND mode for deterministic fixture generation
#                  (default: 2; do not use 4 if you want SEED to matter)
#   STILT_SEED     HYSPLIT SEED value written to SETUP.CFG (default: 42)
#   STILT_MET_DIR  Path to stilt-tutorials/01-wbb/met dir
#                  (default: ../stilt-tutorials/01-wbb/met)
#
# R package dependencies (installed automatically if missing):
#   devtools, uataq (from github.com/uataq/uataq), arrow (for RDS→parquet conversion)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURES_DIR="$SCRIPT_DIR/r_ref"

eval "$(cd "$REPO_ROOT" && uv run python -m tests.fixtures.r_stilt_reference shell)"

STILT_R_DIR="${STILT_R_DIR:-$REPO_ROOT/../R-STILT}"
STILT_TUTORIALS_DIR="${STILT_TUTORIALS_DIR:-$REPO_ROOT/../stilt-tutorials}"
STILT_MET_DIR="${STILT_MET_DIR:-$STILT_TUTORIALS_DIR/01-wbb/met}"
STILT_KRAND="${STILT_KRAND:-$STILT_REFERENCE_KRAND}"
STILT_SEED="${STILT_SEED:-$STILT_REFERENCE_SEED}"

SIM_ID="$STILT_REFERENCE_R_SIM_ID"
OUT_DIR="$STILT_R_DIR/out/by-id/$SIM_ID"
RUN_STILT="$STILT_R_DIR/r/run_stilt.r"
SIMULATION_STEP="$STILT_R_DIR/r/src/simulation_step.r"
WRITE_SETUP="$STILT_R_DIR/r/src/write_setup.r"

echo "=== Generating R-STILT reference fixtures ==="
echo "R-STILT dir : $STILT_R_DIR"
echo "Met dir     : $STILT_MET_DIR"
echo "KRAND       : $STILT_KRAND"
echo "SEED        : $STILT_SEED"
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

# Patch R-STILT sources (creates .bak files for restore on exit)
for file in "$RUN_STILT" "$SIMULATION_STEP" "$WRITE_SETUP"; do
  cp "$file" "$file.bak"
done

cleanup() {
  for file in "$RUN_STILT" "$SIMULATION_STEP" "$WRITE_SETUP"; do
    if [[ -f "$file.bak" ]]; then
      mv "$file.bak" "$file"
      echo "Restored $(basename "$file")"
    fi
  done
}
trap cleanup EXIT

sed -i \
  -e "s|met_path           <- '<path_to_arl_meteorological_data>'|met_path           <- '$STILT_MET_DIR'|" \
  -e "s|t_start <- '2015-12-10 00:00:00'|t_start <- '$STILT_REFERENCE_TIME'|" \
  -e "s|t_end   <- '2015-12-10 00:00:00'|t_end   <- '$STILT_REFERENCE_TIME'|" \
  -e "s|lati <- 40.5|lati <- $STILT_REFERENCE_LATITUDE|" \
  -e "s|long <- -112.0|long <- $STILT_REFERENCE_LONGITUDE|" \
  -e "s|zagl <- 5|zagl <- $STILT_REFERENCE_ALTITUDE|" \
  -e "s|met_file_format    <- '%Y%m%d.%Hz.hrrra'|met_file_format    <- '$STILT_REFERENCE_MET_FILE_FORMAT'|" \
  -e "s|met_file_tres      <- '6 hours'|met_file_tres      <- '$STILT_REFERENCE_R_MET_FILE_TRES'|" \
  -e "s|n_hours       <- -24|n_hours       <- $STILT_REFERENCE_N_HOURS|" \
  -e "s|numpar        <- 1000|numpar        <- $STILT_REFERENCE_NUMPAR|" \
  -e "s|krand       <- 4|krand       <- $STILT_KRAND|" \
  -e "/krand       <- /a seed        <- $STILT_SEED" \
  -e "s|xmn <- NA|xmn <- $STILT_REFERENCE_XMIN|" \
  -e "s|xmx <- NA|xmx <- $STILT_REFERENCE_XMAX|" \
  -e "s|ymn <- NA|ymn <- $STILT_REFERENCE_YMIN|" \
  -e "s|ymx <- NA|ymx <- $STILT_REFERENCE_YMAX|" \
  -e "s|xres <- 0.01|xres <- $STILT_REFERENCE_XRES|" \
  -e "s|yres <- xres|yres <- $STILT_REFERENCE_YRES|" \
  -e "s|krand = krand,|krand = krand,\n            seed = seed,|" \
  "$RUN_STILT"

sed -i \
  -e "s|krand = 4,|krand = 4,\n                            seed = NULL,|" \
  -e "s|krand = krand,|krand = krand,\n      seed = seed,|" \
  "$SIMULATION_STEP"

sed -i \
  -e "s|krand = 4,|krand = 4,\n                        seed = NULL,|" \
  -e "/if (is.logical(rhs))/i\    if (is.null(rhs))\n\      return(NULL)" \
  -e "s|eq('KRAND', krand),|eq('KRAND', krand),\n           eq('SEED', seed),|" \
  "$WRITE_SETUP"

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
cp "$TRAJ_PARQUET" "$FIXTURES_DIR/r_traj.parquet"
cp "$FOOT_NC"      "$FIXTURES_DIR/r_foot.nc"

echo ""
echo "=== Fixtures written ==="
ls -lh "$FIXTURES_DIR"
echo ""
echo "Commit these files to lock the reference:"
echo "  git add tests/fixtures/r_ref/"
echo "  git commit -m 'test: update R-STILT reference fixtures'"
