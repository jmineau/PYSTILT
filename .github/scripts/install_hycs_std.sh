#!/usr/bin/env bash
# Install STILT-R's Linux hycs_std, at the commit STILT_R_SHA in
# .github/workflows/stilt-r.yml pins, into /usr/local/bin. PYSTILT does not
# include HYSPLIT, so the jobs that run it take the build STILT-R's parity is
# checked against.
set -euo pipefail
sha="$(sed -nE 's/^ *STILT_R_SHA: *([0-9a-f]{40}).*/\1/p' .github/workflows/stilt-r.yml)"
if [ -z "$sha" ]; then
  echo "::error::No STILT_R_SHA in .github/workflows/stilt-r.yml."
  exit 1
fi
tmp="$(mktemp)"
curl -fsSL --retry 3 -o "$tmp" \
  "https://raw.githubusercontent.com/uataq/stilt/$sha/bin/linux_x64/hycs_std"
sudo install -m 755 "$tmp" /usr/local/bin/hycs_std
rm -f "$tmp"
echo "hycs_std of uataq/stilt@$sha: $(sha256sum < /usr/local/bin/hycs_std)"
