#!/usr/bin/env bash
# Install STILT-R's Linux HYSPLIT, at the commit STILT_R_SHA in
# .github/workflows/stilt-r.yml pins, as a HYSPLIT folder: /opt/hysplit/exec/
# hycs_std and the four data tables in /opt/hysplit/bdyfiles/, with hycs_std on
# PATH. PYSTILT includes no HYSPLIT, so the jobs that run it take the build
# STILT-R's parity is checked against.
set -euo pipefail
sha="$(sed -nE 's/^ *STILT_R_SHA: *([0-9a-f]{40}).*/\1/p' .github/workflows/stilt-r.yml)"
if [ -z "$sha" ]; then
  echo "::error::No STILT_R_SHA in .github/workflows/stilt-r.yml."
  exit 1
fi
base="https://raw.githubusercontent.com/uataq/stilt/$sha"
tmp="$(mktemp -d)"
mkdir -p "$tmp/exec" "$tmp/bdyfiles"
curl -fsSL --retry 3 -o "$tmp/exec/hycs_std" "$base/bin/linux_x64/hycs_std"
for table in ASCDATA.CFG LANDUSE.ASC ROUGLEN.ASC TERRAIN.ASC; do
  curl -fsSL --retry 3 -o "$tmp/bdyfiles/$table" "$base/exe/$table"
done
chmod +x "$tmp/exec/hycs_std"
sudo mkdir -p /opt
sudo mv "$tmp" /opt/hysplit
sudo ln -sf /opt/hysplit/exec/hycs_std /usr/local/bin/hycs_std
echo "HYSPLIT of uataq/stilt@$sha in /opt/hysplit:"
(cd /opt/hysplit && sha256sum exec/hycs_std bdyfiles/*)
