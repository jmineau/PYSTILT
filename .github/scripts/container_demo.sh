#!/usr/bin/env bash
# Run one release of a small project in the container image, as a user would:
# a project folder with config.yaml and releases.csv, the met files and a
# HYSPLIT folder mounted, then `stilt run`. It passes when the release has its
# particles, its footprint and its map.
#
#   .github/scripts/container_demo.sh IMAGE MET_DIR HYSPLIT_DIR
set -euo pipefail
image="$1"
met="$(realpath "$2")"
hysplit="$(realpath "$3")"
project="$(mktemp -d)"
chmod 777 "$project"  # the image runs as its own user

cat > "$project/config.yaml" <<'EOF'
mets:
  hrrr: /met
models:
  hysplit: {n_hours: -12, numpar: 100, krand: 2, seed: 42}
stages:
  footprint:
    class: stilt.footprint.FootprintStage
    grid: {xmin: -113.0, xmax: -110.5, ymin: 40.0, ymax: 42.0, xres: 0.01, yres: 0.01}
  map:
    class: stilt.footprint.MapStage
variants:
  hrrr: {}
EOF
cat > "$project/releases.csv" <<'EOF'
time,longitude,latitude,altitude
2021-01-15 10:00,-111.848,40.766,10
EOF

docker run --rm \
  -v "$hysplit:/opt/hysplit:ro" \
  -v "$met:/met:ro" \
  -v "$project:/project" \
  "$image" sh -c 'stilt check /project && stilt run /project && stilt status /project'

for kind in particles footprint map; do
  found="$(find "$project" -path "*/$kind/*" -name '202101151000_*' -type f | head -n 1)"
  if [ -z "$found" ]; then
    echo "::error::The release has no $kind file."
    exit 1
  fi
  echo "$kind: ${found#"$project"/}"
done
python3 - "$(find "$project" -path '*/map/*' -name '*.json' | head -n 1)" <<'EOF'
import json, sys
entry = json.load(open(sys.argv[1]))
assert entry["image"].startswith("data:image/png;base64,"), "the map has no image"
print("map bounds:", entry["bounds"])
EOF
