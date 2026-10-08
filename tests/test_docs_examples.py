"""The examples in the docs must be code and settings PYSTILT accepts."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from stilt.config import ProjectConfig
from stilt.execution.config import ExecutionConfig

DOCS = Path(__file__).parent.parent / "docs"
README = DOCS.parent / "README.md"

#: Settings a config example leaves out because the page is about something
#: else: one met, one variant, and a grid, so a fragment such as a
#: ``transforms:`` list validates as part of a whole config.
_BASE_CONFIG = {
    "mets": {
        "hrrr": {"directory": "/met", "file_format": "%Y%m%d_%H", "file_tres": "1h"}
    },
    "variants": {"hrrr": {}},
    "grid": {
        "xmin": -114.0,
        "xmax": -113.0,
        "ymin": 39.0,
        "ymax": 40.0,
        "xres": 0.1,
        "yres": 0.1,
    },
}

#: YAML blocks that are not a ``config.yaml``: a Kubernetes manifest, and
#: transforms from a placeholder package that cannot be imported.
_NOT_A_CONFIG = ("apiVersion:", "mypkg.")


def _pages() -> list[Path]:
    pages = [README]
    for path in sorted(DOCS.rglob("*.rst")):
        if "_build" not in path.parts and "_api" not in path.parts:
            pages.append(path)
    return pages


def _blocks(path: Path, language: str) -> list[str]:
    """Return the code blocks in one language of a reStructuredText or Markdown file."""
    text = path.read_text()
    if path.suffix == ".md":
        return re.findall(rf"```{language}\n(.*?)```", text, flags=re.S)
    # A directive's options (":caption: ...") come before the blank line.
    pattern = rf"\.\. code-block:: {language}\n(?:[ \t]+:.*\n)*\n((?:[ \t]+.*\n|\n)+)"
    return [textwrap.dedent(m.group(1)) for m in re.finditer(pattern, text)]


def _examples(language: str) -> list[tuple[str, str]]:
    return [
        (f"{path.relative_to(DOCS.parent)}#{i}", block)
        for path in _pages()
        for i, block in enumerate(_blocks(path, language))
    ]


def _settings(blocks: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    found = []
    for name, block in blocks:
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict):
            found.append((name, data))
    return found


PYTHON = _examples("python")
YAML = _settings(_examples("yaml"))
CONFIGS = [
    (name, data)
    for name, data in YAML
    if not any(s in yaml.safe_dump(data) for s in _NOT_A_CONFIG)
    # A settings folder's _settings.yaml, on the output layout page.
    and not {"hash", "settings"} <= data.keys()
]
EXECUTION = [
    (name, data["execution"])
    for name, data in YAML
    if isinstance(data.get("execution"), dict)
]


def test_the_docs_show_examples():
    assert len(PYTHON) >= 50
    assert len(CONFIGS) >= 20
    assert len(EXECUTION) >= 3


@pytest.mark.parametrize("block", [b for _, b in PYTHON], ids=[n for n, _ in PYTHON])
def test_python_examples_in_the_docs_compile(block):
    # An interactive session belongs in a ``pycon`` block, which is not compiled.
    compile(block, "<docs>", "exec")


@pytest.mark.parametrize("data", [d for _, d in CONFIGS], ids=[n for n, _ in CONFIGS])
def test_config_examples_in_the_docs_are_valid(data):
    ProjectConfig.model_validate({**_BASE_CONFIG, **data})


@pytest.mark.parametrize(
    "execution", [e for _, e in EXECUTION], ids=[n for n, _ in EXECUTION]
)
def test_execution_examples_in_the_docs_are_valid(execution):
    ExecutionConfig.model_validate(execution)
