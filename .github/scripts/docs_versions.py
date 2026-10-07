"""
Publish one docs build into the versioned GitHub Pages site.

The site (the gh-pages branch) holds one folder per published version::

    dev/          the main branch
    0.2.0/        each release, by version
    stable/       a copy of the newest final release
    switcher.json the version dropdown's entries
    index.html    redirects to stable/ (or dev/ before the first release)
    404.html      sends links from before versioning to the same page in stable/

Run by the Documentation workflow: ``python docs_versions.py SITE HTML TARGET``,
where TARGET is ``dev`` or a version such as ``0.2.0``.
"""

import argparse
import json
import shutil
from pathlib import Path

from packaging.version import InvalidVersion, Version


def _version(name: str) -> Version | None:
    try:
        return Version(name)
    except InvalidVersion:
        return None


def main() -> None:
    """Copy the build into place, then rewrite the switcher and redirects."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", type=Path, help="checkout of the gh-pages branch")
    parser.add_argument("html", type=Path, help="the new HTML build")
    parser.add_argument("target", help='"dev" or the release version')
    parser.add_argument("--base-url", required=True, help="the site's URL, ending in /")
    parser.add_argument(
        "--prereleases",
        choices=["upcoming", "all", "none"],
        default="upcoming",
        help=(
            "upcoming: list pre-releases newer than the latest final release and "
            "delete older ones; all: list every pre-release; none: list none"
        ),
    )
    args = parser.parse_args()
    site, base = args.site, args.base_url

    if args.target != "dev" and _version(args.target) is None:
        parser.error(f"{args.target!r} is neither 'dev' nor a PEP 440 version")
    shutil.rmtree(site / args.target, ignore_errors=True)
    shutil.copytree(args.html, site / args.target)

    # Every version folder, newest first. Folder names are the tags without "v".
    versions = sorted(
        ((d.name, v) for d in site.iterdir() if d.is_dir() and (v := _version(d.name))),
        key=lambda item: item[1],
        reverse=True,
    )
    finals = [name for name, v in versions if not v.is_prerelease]
    latest_final = _version(finals[0]) if finals else None

    def listed(v: Version) -> bool:
        if not v.is_prerelease or args.prereleases == "all":
            return True
        if args.prereleases == "none":
            return False
        return latest_final is None or v > latest_final  # "upcoming"

    if args.prereleases == "upcoming":
        for name, v in versions:
            if not listed(v):
                shutil.rmtree(site / name)  # superseded by a final release
        versions = [(name, v) for name, v in versions if listed(v)]
    shown = [name for name, v in versions if listed(v)]

    # stable/ is the newest final release; before the first one, the newest of any.
    stable = finals[0] if finals else (versions[0][0] if versions else None)
    shutil.rmtree(site / "stable", ignore_errors=True)
    if stable:
        shutil.copytree(site / stable, site / "stable")

    entries = []
    if (site / "dev").is_dir():
        entries.append({"name": "dev", "version": "dev", "url": f"{base}dev/"})
    for name in shown:
        if name == stable:
            label = "latest" if Version(name).is_prerelease else "stable"
            entries.append(
                {
                    "name": f"{name} ({label})",
                    "version": name,
                    "url": f"{base}stable/",
                    "preferred": True,
                }
            )
        else:
            entries.append({"name": name, "version": name, "url": f"{base}{name}/"})
    (site / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")

    home = "stable" if stable else "dev"
    (site / "index.html").write_text(
        f'<!doctype html>\n<meta charset="utf-8">\n<title>Redirecting</title>\n'
        f'<meta http-equiv="refresh" content="0; url={home}/">\n'
        f'<link rel="canonical" href="{base}{home}/">\n<a href="{home}/">Go to the documentation</a>\n'
    )
    known = json.dumps(["dev", "stable", *(name for name, _ in versions)])
    (site / "404.html").write_text(
        f"""<!doctype html>
<meta charset="utf-8">
<title>Page not found</title>
<script>
  // A link from before the docs were versioned (e.g. .../api.html): retry it in {home}/.
  const root = new URL("{base}").pathname;
  const rest = location.pathname.startsWith(root) ? location.pathname.slice(root.length) : "";
  if (rest && !{known}.includes(rest.split("/")[0])) {{
    location.replace(root + "{home}/" + rest + location.search + location.hash);
  }}
</script>
<p>Page not found. <a href="{base}">Go to the documentation</a>.</p>
"""
    )


if __name__ == "__main__":
    main()
