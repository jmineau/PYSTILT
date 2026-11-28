"""
The ``stilt`` command: the particle commands, and ``stilt init``, which starts a STILT project.

The commands that run a project and report on it (``check``, ``fetch``,
``run``, ``submit``, ``status``) are :mod:`runtraj.cli`'s, which open the
particle part of a project. ``stilt init`` is STILT's: it writes the
starter config, with STILT-R's HYSPLIT values and the footprint stage
(:meth:`stilt.Project.init`). Examples::

    stilt init                        # start a project in the current directory
    stilt init ./my_project           # start a project in ./my_project

The exit codes are :mod:`runtraj.cli`'s.
"""

from __future__ import annotations

from pathlib import Path

import typer
from runtraj import cli

from stilt.project import Project

app = typer.Typer(
    name="stilt",
    help="Run STILT simulations and check on a project.",
    no_args_is_help=True,
    cls=cli.ExitCodeGroup,
)

_NEW_PROJECT_ARG = typer.Argument(
    None,
    help="Directory for the new project. Defaults to the current directory.",
)


@app.command()
def init(project: Path = _NEW_PROJECT_ARG) -> None:
    """
    Create a new project with a starter config.yaml and releases.csv.

    Edit both files before running stilt run. Stops if the directory
    already has a config.yaml.
    """
    project = (project or Path.cwd()).resolve()
    try:
        Project.init(project, starter=True)
    except FileExistsError:
        typer.echo(
            f"Error: '{project}' already contains a config.yaml. Aborting.",
            err=True,
        )
        raise typer.Exit(code=cli.EXIT_USAGE) from None

    typer.echo(f"Initialized STILT project at '{project}'")
    typer.echo(
        "  config.yaml    edit the met directory and the footprint grid (a placeholder over Utah)"
    )
    typer.echo("  releases.csv  add release times and locations, one per line,")
    typer.echo("                as in this one, which is in Salt Lake City:")
    typer.echo("                  2023-01-01 12:00:00,-111.85,40.77,5")


# The particle commands, after init, in the order runtraj.cli registers them.
app.registered_commands.extend(cli.app.registered_commands)


__all__ = ["app"]
