"""Module entry point for ``python -m stilt``."""

from stilt.cli import app


def main() -> None:
    """Run the Typer CLI."""
    app()


if __name__ == "__main__":
    main()
