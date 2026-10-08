"""The examples in the docs must be code and settings PYSTILT accepts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
import types
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

#: YAML blocks that are not a ``config.yaml``: a Kubernetes manifest.
_NOT_A_CONFIG = ("apiVersion:",)

#: The first line of the docs' block that defines the placeholder package
#: ``mypkg.transforms``, which a config example then names.
_MYPKG_TRANSFORMS = "# mypkg/transforms.py"


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
    blocks = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        directive = re.fullmatch(rf"(\s*)\.\. code-block:: {language}\s*", line)
        if directive is None:
            continue
        indent = len(directive.group(1))
        body = []
        # The block is every line after the directive indented deeper than
        # it, with its options (":caption: ...") left out.
        for following in lines[i + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        code = textwrap.dedent("\n".join(body)).strip("\n")
        code = re.sub(r"\A(?::\w[\w-]*:.*\n)*", "", code)
        blocks.append(code + "\n")
    return blocks


def _examples(language: str) -> list[tuple[str, str]]:
    return [
        (f"{path.relative_to(DOCS.parent)}#{i}", block)
        for path in _pages()
        for i, block in enumerate(_blocks(path, language))
    ]


def _settings(blocks: list[tuple[str, str]]) -> list[tuple[str, dict]]:
    found = []
    for name, block in blocks:
        data = yaml.safe_load(block)
        if isinstance(data, dict):
            found.append((name, data))
    return found


def _statements(session: str) -> str:
    """Return the code of an interactive session (``pycon``), without its prompts and output."""
    return "\n".join(
        line[4:] for line in session.splitlines() if line.startswith((">>> ", "... "))
    )


PYTHON = _examples("python")
PYCON = _examples("pycon")
BASH = _examples("bash")
YAML_BLOCKS = _examples("yaml")


@pytest.mark.parametrize(
    "block", [b for _, b in YAML_BLOCKS], ids=[n for n, _ in YAML_BLOCKS]
)
def test_yaml_examples_in_the_docs_parse(block):
    yaml.safe_load(block)


def _parsed(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return the blocks that parse; the test above fails the others."""
    good = []
    for name, block in blocks:
        try:
            yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        good.append((name, block))
    return good


YAML = _settings(_parsed(YAML_BLOCKS))
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
    assert len(BASH) >= 10
    assert PYCON


@pytest.fixture
def mypkg(monkeypatch):
    """Make the docs' placeholder ``mypkg.transforms`` importable, from the docs' own block."""
    source = next(b for _, b in PYTHON if b.startswith(_MYPKG_TRANSFORMS))
    package = types.ModuleType("mypkg")
    package.__path__ = []
    module = types.ModuleType("mypkg.transforms")
    exec(compile(source, "mypkg/transforms.py", "exec"), module.__dict__)
    package.transforms = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mypkg", package)
    monkeypatch.setitem(sys.modules, "mypkg.transforms", module)


@pytest.mark.parametrize("block", [b for _, b in PYTHON], ids=[n for n, _ in PYTHON])
def test_python_examples_in_the_docs_compile(block):
    compile(block, "<docs>", "exec")


@pytest.mark.parametrize("block", [b for _, b in PYCON], ids=[n for n, _ in PYCON])
def test_interactive_examples_in_the_docs_compile(block):
    compile(_statements(block), "<docs>", "exec")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("block", [b for _, b in BASH], ids=[n for n, _ in BASH])
def test_shell_examples_in_the_docs_are_valid_bash(block):
    checked = subprocess.run(
        ["bash", "-n"], input=block, text=True, capture_output=True
    )
    assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("data", [d for _, d in CONFIGS], ids=[n for n, _ in CONFIGS])
def test_config_examples_in_the_docs_are_valid(data, mypkg):
    ProjectConfig.model_validate({**_BASE_CONFIG, **data})


@pytest.mark.parametrize(
    "execution", [e for _, e in EXECUTION], ids=[n for n, _ in EXECUTION]
)
def test_execution_examples_in_the_docs_are_valid(execution):
    ExecutionConfig.model_validate(execution)
