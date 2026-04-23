"""Sphinx helpers for rendering PYSTILT config model tables."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import stringify_annotation


def _resolve_model(path: str) -> type[BaseModel]:
    """Import a dotted model path and validate that it is a Pydantic model."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a dotted path, got {path!r}")
    obj = getattr(import_module(module_name), attr)
    if not isinstance(obj, type) or not issubclass(obj, BaseModel):
        raise TypeError(f"{path!r} is not a Pydantic BaseModel subclass")
    return obj


def _resolve_object(path: str) -> Any:
    """Import a dotted object path."""
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected a dotted path, got {path!r}")
    return getattr(import_module(module_name), attr)


def _field_visibility(field: Any) -> str:
    """Return a normalized visibility label from field metadata."""
    return str((field.json_schema_extra or {}).get("visibility", "public"))


def _iter_fields(
    model: type[BaseModel],
    *,
    declared_only: bool,
    include_internal: bool,
) -> list[tuple[str, Any]]:
    """Return fields in declaration order, optionally excluding inherited ones."""
    declared = set(getattr(model, "__annotations__", {}))
    rows: list[tuple[str, Any]] = []
    for name, field in model.model_fields.items():
        if declared_only and name not in declared:
            continue
        if _field_visibility(field) == "internal" and not include_internal:
            continue
        rows.append((name, field))
    return rows


def _format_type(annotation: Any) -> str:
    """Render a concise type string suitable for docs tables."""
    try:
        rendered = stringify_annotation(annotation, mode="smart")
    except TypeError:
        rendered = stringify_annotation(annotation)
    return rendered.replace("stilt.config.", "")


def _format_default(field: Any) -> str:
    """Render a field default value for docs tables."""
    if field.default_factory is not None:
        try:
            value = field.default_factory()
        except Exception:
            return "``<factory>``"
    elif field.default is PydanticUndefined:
        return "Required"
    else:
        value = field.default

    rendered = repr(value)
    if len(rendered) > 88:
        rendered = f"{rendered[:85]}..."
    return f"``{rendered}``"


def _format_description(field: Any) -> str:
    """Normalize a field description into a single readable paragraph."""
    description = " ".join((field.description or "").split())
    description = description.replace("\\", "\\\\").replace("*", "\\*")
    return description or "-"


def _docstring_summary(obj: Any) -> list[str]:
    """Return the opening narrative block of an object's docstring."""
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return []

    lines = doc.splitlines()
    kept: list[str] = []
    for i, line in enumerate(lines):
        if (
            i + 1 < len(lines)
            and line.strip()
            and set(lines[i + 1].strip()) == {"-"}
            and len(lines[i + 1].strip()) >= len(line.strip())
        ):
            break
        kept.append(line)

    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _docstring_section(obj: Any, section: str) -> list[str]:
    """Return one NumPy-style docstring section body by heading name."""
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return []

    lines = doc.splitlines()
    section_start: int | None = None
    section_end = len(lines)

    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if not title or not underline or set(underline) != {"-"}:
            continue
        if len(underline) < len(title):
            continue
        if title == section:
            section_start = i + 2
            continue
        if section_start is not None:
            section_end = i
            break

    if section_start is None:
        return []

    body = lines[section_start:section_end]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return body


def _format_parameter_section(lines: list[str]) -> list[str]:
    """Convert NumPy-style parameter headers into a clean definition list."""
    formatted: list[str] = []
    for line in lines:
        if line.strip() and not line.startswith((" ", "\t")):
            if " : " in line:
                name, annotation = line.split(" : ", 1)
                formatted.append(f"``{name}`` : {annotation}")
            else:
                formatted.append(f"``{line}``")
            continue
        formatted.append(line)
    return formatted


class ConfigModelDirective(SphinxDirective):
    """Render a model's config fields as a readable list-table."""

    required_arguments = 1
    has_content = False
    option_spec = {
        "declared-only": directives.flag,
        "include-internal": directives.flag,
    }

    def run(self) -> list[nodes.Node]:
        model = _resolve_model(self.arguments[0])
        rows = _iter_fields(
            model,
            declared_only="declared-only" in self.options,
            include_internal="include-internal" in self.options,
        )

        source = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 22 60 18",
            "",
            "   * - Parameter",
            "     - Description",
            "     - Default",
        ]

        for name, field in rows:
            source.extend(
                [
                    f"   * - ``{name}``",
                    f"     - {_format_description(field)}",
                    f"     - {_format_default(field)}",
                ]
            )

        content = StringList(source)
        container = nodes.container()
        self.state.nested_parse(content, self.content_offset, container)
        return container.children


class ClassSignatureDirective(SphinxDirective):
    """Render a class signature with a short summary, without full autodoc body."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        obj = _resolve_object(self.arguments[0])
        module_name, _, attr = self.arguments[0].rpartition(".")
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = "()"

        source = [f".. py:class:: {attr}{signature}"]
        if module_name:
            source.extend([f"   :module: {module_name}", ""])
        else:
            source.append("")
        for line in _docstring_summary(obj):
            source.append(f"   {line}" if line else "")

        content = StringList(source)
        container = nodes.container()
        self.state.nested_parse(content, self.content_offset, container)
        return container.children


class ClassParametersDirective(SphinxDirective):
    """Render the raw NumPy-style Parameters section for one class."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        section = _docstring_section(_resolve_object(self.arguments[0]), "Parameters")
        if not section:
            return []

        content = StringList(
            [".. rubric:: Parameters", "", *_format_parameter_section(section)]
        )
        container = nodes.container()
        self.state.nested_parse(content, self.content_offset, container)
        return container.children


def setup(app: Any) -> dict[str, bool]:
    """Register the custom directive."""
    app.add_directive("config-model", ConfigModelDirective)
    app.add_directive("class-signature", ClassSignatureDirective)
    app.add_directive("class-parameters", ClassParametersDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
