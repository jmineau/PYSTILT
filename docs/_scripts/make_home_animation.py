"""
Make the animation on the docs home page.

The animation shows one PYSTILT simulation of the WLEF tall tower in Park
Falls, Wisconsin (NOAA GML site LEF, inlet 396 m above ground) at 20:00 UTC
on 8 February 2019. On the left, the particles travel backward in time from
the tower. On the right, the footprint adds up hour by hour as they go.

The docs build does not run this script. Run it by hand in three steps, from
the repository root, with the ``dev`` environment (``just sync``)::

    uv run python docs/_scripts/make_home_animation.py init <project> --met-dir <dir>
    uv run stilt run <project>
    uv run python docs/_scripts/make_home_animation.py render <project>

``init`` writes the project. ``--met-dir`` is a folder of NOAA ARL's HRRR
files that covers 7 and 8 February 2019; without it, the project downloads
them from NOAA (five 6-hour files, about 3.4 GB each). The run takes a few
minutes; on a shared cluster run it in a job. ``render`` draws the frames on
USGS map tiles (it downloads them) and writes
``docs/_static/home-animation.mp4`` and ``home-animation-poster.jpg``. It
needs an ``ffmpeg`` built with libx264 (set ``FFMPEG`` to use another one than
the one on ``PATH``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import stilt

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "_static"

#: The WLEF tower (ObsPack site LEF) and the time of the simulation.
TOWER_LON, TOWER_LAT, INLET_M = -90.2732, 45.9451, 396.0
TIME = "2019-02-08 20:00"

#: The map: from the tower north to the edge of HRRR's domain (about 52.6 N
#: here), where particles leave the meteorology and stop.
EXTENT = (-99.0, -84.5, 44.6, 52.6)

#: Minutes of backward time between frames, frames per second, and how many
#: seconds the last frame is held before the video loops.
STEP_MIN = 10
FPS = 16
HOLD_S = 2.5

#: USGS map tiles: shaded relief under the water layer. Both are public domain.
RELIEF = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}"
HYDRO = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSHydroCached/MapServer/tile/{z}/{y}/{x}"
TILE_ZOOM = 7


def init(project: Path, met_dir: Path | None) -> None:
    """Write the project: one receptor, HRRR, 1000 particles, 24 hours back."""
    crop = {"xmin": -105.0, "xmax": -75.0, "ymin": 38.0, "ymax": 56.0}
    if met_dir is None:
        met = {
            "directory": "met",
            "download": "hrrr",
            "subgrid_enable": True,
            "subgrid_bounds": crop,
        }
    else:
        met = {
            "directory": str(met_dir),
            "subgrid_enable": True,
            "subgrid_bounds": crop,
            "subgrid_dir": "met_crop",
        }
    receptor = stilt.PointReceptor(
        time=pd.Timestamp(TIME),
        longitude=TOWER_LON,
        latitude=TOWER_LAT,
        altitude=INLET_M,
    )
    stilt.Project.init(
        project,
        receptors=[receptor],
        mets={"hrrr": met},
        n_hours=-24,
        numpar=1000,
        krand=2,
        seed=42,
        grid={
            "xmin": EXTENT[0],
            "xmax": EXTENT[1],
            "ymin": EXTENT[2],
            "ymax": EXTENT[3],
            "xres": 0.025,
            "yres": 0.025,
        },
        tres="1h",
        variants={"hrrr": {}},
    )
    print(f"Wrote {project}. Next: stilt run {project}")


def tiles(url: str, cache: Path):
    """Return a cartopy tiler for *url* that keeps each tile it downloads in *cache*."""
    from urllib.request import Request, urlopen

    from cartopy.io.img_tiles import GoogleTiles
    from PIL import Image

    class CachedTiles(GoogleTiles):
        def get_image(self, tile):
            x, y, z = tile
            path = cache / str(z) / str(y) / f"{x}.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                request = Request(
                    self._image_url(tile), headers={"User-Agent": "pystilt-docs"}
                )
                with urlopen(request, timeout=60) as response:
                    path.write_bytes(response.read())
            img = Image.open(path).convert("RGBA")
            return img, self.tileextent(tile), "lower"

    return CachedTiles(url=url, desired_tile_form="RGBA")


def render(project: Path) -> None:
    """Draw the frames and write the video and its poster into ``docs/_static``."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image

    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg is None:
        sys.exit("ffmpeg not found: put it on PATH or set FFMPEG.")

    proj = stilt.Project(project)
    sim = proj.simulation(
        proj.simulations["receptor"].iloc[0], proj.simulations["variant"].iloc[0]
    )
    particles = sim.particles
    foot = sim.footprint  # (time, lat, lon), one layer per hour back from the receptor
    receptor_time = pd.Timestamp(TIME)

    # A particle's sensitivity and a cell's footprint are different
    # quantities, so each gets its own scale, from its largest value down
    # three orders of magnitude; the color bar says only low and high.
    top = float(foot.sum("time").max())
    norm = LogNorm(vmin=top / 1000, vmax=top)
    sensitive = particles.loc[particles["foot"] > 0, "foot"]
    ptop = float(sensitive.quantile(0.999))
    pnorm = LogNorm(vmin=ptop / 1000, vmax=ptop)
    cmap = plt.get_cmap("plasma")

    cache = Path(tempfile.gettempdir()) / "pystilt-home-animation-tiles"
    relief = tiles(RELIEF, cache / "relief")
    hydro = tiles(HYDRO, cache / "hydro")
    crs = relief.crs

    def base(ax):
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.add_image(relief, TILE_ZOOM, alpha=0.55, zorder=0)
        ax.add_image(hydro, TILE_ZOOM, zorder=1)
        ax.plot(
            TOWER_LON,
            TOWER_LAT,
            "o",
            ms=9,
            mfc="none",
            mec="black",
            mew=1.4,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
        ax.set_axis_off()

    ages = np.arange(0, -24 * 60 - STEP_MIN, -STEP_MIN)
    # Each layer is labeled by its start: 19:00 holds 19:00 to 20:00.
    layer_end = pd.to_datetime(foot["time"].values) + pd.Timedelta(hours=1)
    frames = Path(tempfile.mkdtemp(prefix="home-animation-"))

    # Size the figure to the map, two panels side by side over the color bar.
    x0, y0 = crs.transform_point(EXTENT[0], EXTENT[2], ccrs.PlateCarree())
    x1, y1 = crs.transform_point(EXTENT[1], EXTENT[3], ccrs.PlateCarree())
    width, gap, bar_h = 16.0, 0.08, 0.6
    panel_w = (width - gap) / 2
    panel_h = panel_w * (y1 - y0) / (x1 - x0)
    height = panel_h + bar_h
    fig = plt.figure(figsize=(width, height), dpi=100, facecolor="white")
    rect_y, rect_h, rect_w = bar_h / height, panel_h / height, panel_w / width
    left = fig.add_axes((0, rect_y, rect_w, rect_h), projection=crs)
    right = fig.add_axes((1 - rect_w, rect_y, rect_w, rect_h), projection=crs)
    bar = fig.add_axes((0.25, 0.35 * bar_h / height, 0.5, 0.3 * bar_h / height))
    for ax in (left, right):
        base(ax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=bar, orientation="horizontal")
    cb.set_ticks([])
    cb.minorticks_off()
    cb.outline.set_visible(False)
    fig.text(0.245, 0.5 * bar_h / height, "Low", ha="right", va="center", fontsize=14)
    fig.text(0.755, 0.5 * bar_h / height, "High", ha="left", va="center", fontsize=14)
    fig.text(
        0.995,
        0.003,
        "Map: USGS The National Map",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.4",
    )
    clock = left.text(
        0.02,
        0.975,
        "",
        transform=left.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 4},
        zorder=7,
    )
    right.text(
        0.02,
        0.975,
        "Footprint",
        transform=right.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 4},
        zorder=7,
    )

    lon = foot["lon"].values
    lat = foot["lat"].values
    dynamic = []
    for k, age in enumerate(ages):
        for artist in dynamic:
            artist.remove()
        dynamic = []
        now = particles[particles["age"] == age]
        quiet = now[now["foot"] <= pnorm.vmin]
        loud = now[now["foot"] > pnorm.vmin].sort_values("foot")
        dynamic.append(
            left.scatter(
                quiet["lon"],
                quiet["lat"],
                s=6,
                c="0.35",
                alpha=0.35,
                lw=0,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        )
        dynamic.append(
            left.scatter(
                loud["lon"],
                loud["lat"],
                s=14,
                c=loud["foot"],
                cmap=cmap,
                norm=pnorm,
                alpha=0.8,
                lw=0,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        )
        # The footprint so far: every hourly layer the particles have reached,
        # the one they are in included, so it grows from the first frame.
        reached = layer_end > receptor_time + pd.Timedelta(minutes=int(age))
        if age == 0 or not reached.any():
            so_far = None
        else:
            so_far = foot.isel(time=np.flatnonzero(reached)).sum("time").values
        if so_far is not None:
            shown = np.ma.masked_less_equal(so_far, norm.vmin)
            dynamic.append(
                right.pcolormesh(
                    lon,
                    lat,
                    shown,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.85,
                    shading="nearest",
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            )
        hours = -age / 60
        clock.set_text("Now" if age == 0 else f"{hours:.1f} hours back")
        fig.savefig(frames / f"{k:04d}.png", facecolor="white")
        print(f"frame {k + 1}/{len(ages)}", end="\r")
    print()
    # The poster (shown until the video plays) is the last frame, as a JPEG:
    # the repository refuses files over 500 KB, and the PNG is larger.
    last = frames / f"{len(ages) - 1:04d}.png"
    poster = Image.open(last).convert("RGB")
    poster = poster.resize((1280, round(poster.height * 1280 / poster.width)))
    poster.save(OUT / "home-animation-poster.jpg", quality=85)
    # Hold the last frame before the video loops.
    for k in range(len(ages), len(ages) + int(HOLD_S * FPS)):
        shutil.copy(last, frames / f"{k:04d}.png")

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(frames / "%04d.png"),
            "-vf",
            "scale=1280:-2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "28",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            "-an",
            str(OUT / "home-animation.mp4"),
        ],
        check=True,
    )
    shutil.rmtree(frames)
    for name in ("home-animation.mp4", "home-animation-poster.jpg"):
        print(f"{OUT / name}: {(OUT / name).stat().st_size / 1e6:.1f} MB")


def main() -> None:
    """Run the step named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    sub = parser.add_subparsers(dest="step", required=True)
    p_init = sub.add_parser("init", help="write the project")
    p_init.add_argument("project", type=Path)
    p_init.add_argument(
        "--met-dir", type=Path, default=None, help="folder of NOAA ARL HRRR files"
    )
    p_render = sub.add_parser("render", help="draw the frames and write the videos")
    p_render.add_argument("project", type=Path)
    args = parser.parse_args()
    if args.step == "init":
        init(args.project, args.met_dir)
    else:
        render(args.project)


if __name__ == "__main__":
    main()
