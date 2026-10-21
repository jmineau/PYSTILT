"""
Helpers for the docs pages that run real PYSTILT code when the docs are built.

The pages run on the HRRR files the tests cache: 14 and 15 January 2021,
cropped around Salt Lake. A page that works with a project on disk starts from
a small sample project, which is built once per docs build (about 40 s) and
copied into a temporary folder for each page that uses it.

A page uses these from a block that is run but not shown (``:suppress:``)::

    import docs_examples

    docs_cwd = docs_examples.temp_workdir(sample=True)

and returns to the docs folder at its end with ``os.chdir(docs_cwd)``.

The docs' maps are drawn on USGS shaded-relief tiles kept in ``docs/_tiles``,
so a build needs no network for them and every build draws the same maps
(``docs/conf.py`` calls :func:`use_tiles`). A map whose tiles are not there
fails the build. To add them, build the docs once with
``PYSTILT_DOCS_FETCH_TILES=1``, which downloads each missing tile, and commit
what it added to ``docs/_tiles``.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

DOCS = Path(__file__).resolve().parents[1]

#: Where the sample project is built. ``just build-docs`` clears ``docs/_build``,
#: so every clean build makes a new one; ``just docs-serve`` reuses it.
SAMPLE = DOCS / "_build" / "sample_project"

#: The sample project's receptors: one site, the four hours before 10:00 UTC.
SAMPLE_HOURS = (6, 7, 8, 9)

#: The committed map tiles, one JPEG per tile at ``<z>/<y>/<x>.jpg``.
TILES = DOCS / "_tiles" / "usgs_shaded_relief"

#: Where the tiles come from: the USGS National Map's shaded-relief layer,
#: which needs no key.
TILE_SERVER = (
    "https://basemap.nationalmap.gov/arcgis/rest/services/"
    "USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}"
)

#: The credit the service gives for its tiles, written on each map.
TILE_CREDIT = (
    "Tiles: USGS The National Map, 3D Elevation Program; USGS EROS Center, GMTED2010"
)

#: Set to 1 to download the tiles a build needs and does not have.
FETCH_TILES = "PYSTILT_DOCS_FETCH_TILES"

_temporary_folders: list[Path] = []


def met() -> dict:
    """Return the met config of the files the tests cache."""
    default = DOCS.parent / "tests" / "met_cache"
    directory = Path(os.environ.get("STILT_TEST_MET_DIR", default)).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} is missing. The docs run on the met files the tests "
            "cache: see 'Examples in the docs' in docs/development.rst."
        )
    return {"directory": str(directory), "file_format": "%Y%m%d_%H", "file_tres": "6h"}


def sample_project() -> Path:
    """Return the folder of the sample project, building it the first time."""
    import stilt

    project = SAMPLE / "my_project"
    marker = SAMPLE / "complete"
    if not marker.exists():
        shutil.rmtree(SAMPLE, ignore_errors=True)
        receptors = [
            stilt.PointReceptor(
                time=f"2021-01-15 {hour:02d}:00",
                longitude=-111.848,
                latitude=40.766,
                altitude=10,
            )
            for hour in SAMPLE_HOURS
        ]
        grid = stilt.Grid(
            xmin=-113.0, xmax=-110.5, ymin=40.0, ymax=42.0, xres=0.01, yres=0.01
        )
        stilt.Project.init(
            str(project),
            receptors=receptors,
            mets={"hrrr": met()},
            variants={"hrrr": {}},
            n_hours=-24,
            numpar=200,
            grid=grid,
        ).run()
        marker.touch()
    return project


def _fetch_tile(tile: tuple[int, int, int], path: Path) -> None:
    """Download one tile to ``path`` when the build is asked to, else say how."""
    x, y, z = tile
    if not os.environ.get(FETCH_TILES):
        raise RuntimeError(
            f"The map tile {z}/{y}/{x} is not in {TILES}. Build the docs once "
            f"with {FETCH_TILES}=1 to download it, then commit the new tiles."
        )
    request = Request(
        TILE_SERVER.format(x=x, y=y, z=z), headers={"User-Agent": "PYSTILT docs"}
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
    except OSError as error:
        raise RuntimeError(
            f"Could not download map tile {z}/{y}/{x}: {error}"
        ) from error
    if not data.startswith(b"\xff\xd8"):  # a JPEG, not an error page
        raise RuntimeError(f"Map tile {z}/{y}/{x} is not a JPEG: {data[:80]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def committed_tiler():
    """
    Return a cartopy tiler that reads the committed tiles from ``docs/_tiles``.

    A tile that is missing raises, so the build fails. (cartopy would draw a
    flat grey tile where a tile cannot be read, and say nothing.)
    """
    from cartopy.io.img_tiles import GoogleTiles

    class CommittedTiles(GoogleTiles):
        def get_image(self, tile):
            x, y, z = tile
            path = TILES / str(z) / str(y) / f"{x}.jpg"
            if not path.exists():
                _fetch_tile(tile, path)
            return super().get_image(tile)

    return CommittedTiles(url=f"{TILES.as_uri()}/{{z}}/{{y}}/{{x}}.jpg")


def use_tiles() -> None:
    """
    Draw every new map of this docs build on the committed tiles.

    A map that is given a ``tiler`` keeps it. The service's credit for the
    tiles is written on each new map.
    """
    import stilt.visualization as visualization

    make_ax = visualization._make_ax
    if getattr(make_ax, "on_committed_tiles", False):
        return
    tiles = committed_tiler()

    def make_ax_on_tiles(ax=None, extent=None, tiler=None, tiler_zoom=8):
        fig, new_ax = make_ax(ax, extent, tiler or tiles, tiler_zoom)
        if ax is None and extent is not None:
            new_ax.text(
                0.995,
                0.005,
                TILE_CREDIT,
                transform=new_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 1,
                },
            )
        return fig, new_ax

    make_ax_on_tiles.on_committed_tiles = True
    visualization._make_ax = make_ax_on_tiles


def temp_workdir(sample: bool = False) -> str:
    """
    Change to a new temporary folder and return the folder you left.

    With ``sample``, the new folder holds a copy of the sample project as
    ``my_project``, so a page can open it the way a reader would.
    """
    previous = os.getcwd()
    work = Path(tempfile.mkdtemp())
    _temporary_folders.append(work)
    if sample:
        shutil.copytree(sample_project(), work / "my_project")
    os.chdir(work)
    return previous


@atexit.register
def _remove_temporary_folders() -> None:
    for folder in _temporary_folders:
        shutil.rmtree(folder, ignore_errors=True)
