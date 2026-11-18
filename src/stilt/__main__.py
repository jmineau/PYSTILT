"""Module entry point for ``python -m stilt``."""

from stilt.stilt_cli import app


def main() -> None:
    """Run the Typer CLI."""
    app()


if __name__ == "__main__":
    main()
