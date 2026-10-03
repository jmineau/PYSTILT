"""The ``execution:`` examples in the docs must be settings PYSTILT accepts."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml

from stilt.execution.config import ExecutionConfig

DOCS = Path(__file__).parent.parent / "docs"
README = DOCS.parent / "README.md"


def _yaml_blocks(path: Path) -> list[str]:
    """Return the YAML code blocks of one reStructuredText or Markdown file."""
    text = path.read_text()
    if path.suffix == ".md":
        return re.findall(r"```yaml\n(.*?)```", text, flags=re.S)
    blocks = []
    for match in re.finditer(r"\.\. code-block:: yaml\n\n((?:[ \t]+.*\n|\n)+)", text):
        blocks.append(textwrap.dedent(match.group(1)))
    return blocks


def _execution_examples() -> list[tuple[str, dict]]:
    found = []
    for path in [README, *sorted(DOCS.rglob("*.rst"))]:
        if "_build" in path.parts or "_api" in path.parts:
            continue
        for block in _yaml_blocks(path):
            try:
                data = yaml.safe_load(block)
            except yaml.YAMLError:
                continue
            if isinstance(data, dict) and isinstance(data.get("execution"), dict):
                found.append((str(path.relative_to(DOCS.parent)), data["execution"]))
    return found


EXAMPLES = _execution_examples()


def test_the_docs_show_execution_examples():
    assert len(EXAMPLES) >= 3


@pytest.mark.parametrize(
    "execution",
    [e for _, e in EXAMPLES],
    ids=[f"{p}#{i}" for i, (p, _) in enumerate(EXAMPLES)],
)
def test_execution_examples_in_the_docs_are_valid(execution):
    ExecutionConfig.model_validate(execution)
