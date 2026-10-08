"""
Class pages for the API reference, in the style of pandas'.

A class page (``_templates/autosummary/class.rst``) shows the class
docstring, then a table of its attributes and one of its methods. Each row
links to the member's own page. This extension decides what goes in the
tables, and keeps the class docstring from describing it a second time.

A member gets a row and a page when the package defines it (not when a
pydantic model inherits it from ``BaseModel``, or an exception from
``Exception``) and it is public. A member a subclass inherits from
another class of the package is a link to that class's page for it, so
document the base classes too (a base with no page leaves plain names).

A member without a docstring of its own (a ``#:`` comment or a string
after it, for data) gets no row when the class docstring describes it
instead, in its Parameters or Attributes section: a dataclass field, an
attribute set in ``__init__``, a property documented only there. Its page
would be empty. The Attributes entries of the members that do get a row
are dropped, so nothing is described twice.
"""

import enum
import functools
import inspect

from sphinx.errors import PycodeError
from sphinx.ext.autosummary import import_by_name
from sphinx.pycode import ModuleAnalyzer

ROUTINES = (property, functools.cached_property, classmethod, staticmethod)


def _package(obj: object) -> str:
    return (getattr(obj, "__module__", None) or "").split(".")[0]


def _owner(cls: type, name: str) -> type | None:
    """Return the class in *cls*'s MRO that defines *name*, if any."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
        try:
            if name in inspect.get_annotations(klass):
                return klass
        except Exception:  # an annotation that cannot be evaluated
            continue
    return None


def _defined_here(cls: type, name: str) -> bool:
    """
    Whether *cls*'s package defines its member *name*.

    A method that overrides a method of a base class from another package
    counts. A data attribute that does, such as a pydantic model's
    ``model_config``, does not.
    """
    owner = _owner(cls, name)
    if owner is None or _package(owner) != _package(cls):
        return False
    value = vars(owner).get(name)
    if callable(value) or isinstance(value, ROUTINES):
        return True
    return not any(name in vars(b) for b in cls.__mro__ if _package(b) != _package(cls))


@functools.cache
def _attr_docs(module: str) -> dict[tuple[str, str], list[str]]:
    """Return the ``#:`` comments and docstrings of a module's attributes."""
    try:
        return ModuleAnalyzer.for_module(module).find_attr_docs()
    except PycodeError:
        return {}


def _has_own_doc(cls: type, name: str) -> bool:
    """Whether member *name* has a docstring (or a ``#:`` comment) of its own."""
    owner = _owner(cls, name)
    if owner is None:
        return False
    value = vars(owner).get(name)
    if callable(value) or isinstance(value, ROUTINES):
        return bool(getattr(value, "__doc__", None))
    return (owner.__qualname__, name) in _attr_docs(owner.__module__)


def _is_routine(cls: type, name: str) -> bool:
    """Whether member *name* is a method or property, rather than data."""
    owner = _owner(cls, name)
    value = vars(owner).get(name) if owner else None
    return callable(value) or isinstance(value, ROUTINES)


def _is_underline(line: str) -> bool:
    return bool(line.strip()) and set(line.strip()) == {"-"}


def _section(lines: list[str], title: str) -> tuple[int, int] | None:
    """Start and end of the NumPy section *title* in *lines*, if it has one."""
    for i in range(len(lines) - 1):
        if lines[i].strip() == title and _is_underline(lines[i + 1]):
            for end in range(i + 2, len(lines) - 1):
                if lines[end].strip() and _is_underline(lines[end + 1]):
                    return i, end
            return i, len(lines)
    return None


def _entries(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split a section's body into (name, lines) entries."""
    indent = min((len(x) - len(x.lstrip()) for x in lines if x.strip()), default=0)
    entries: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip()) == indent:
            entries.append((line.split(":")[0].strip(), [line]))
        elif entries:
            entries[-1][1].append(line)
    return entries


@functools.cache
def _described_attributes(cls: type) -> frozenset[str]:
    """Return the names in the Attributes sections of *cls* and its bases."""
    names = set()
    for klass in cls.__mro__:
        lines = inspect.cleandoc(vars(klass).get("__doc__") or "").splitlines()
        if span := _section(lines, "Attributes"):
            names.update(name for name, _ in _entries(lines[span[0] + 2 : span[1]]))
    return frozenset(names)


def _has_page(cls: type, name: str) -> bool:
    """Whether the class page lists *name* in a table, with its own page."""
    if name.startswith("_") and name != "__call__" or not _defined_here(cls, name):
        return False
    if _has_own_doc(cls, name):
        return True
    # Undocumented: a row only for a method or property the class docstring
    # does not describe either.
    return _is_routine(cls, name) and name not in _described_attributes(cls)


def _link_name(cls: type) -> str | None:
    """
    Return the name to link to a base class by, or None if it has no public name.

    This is the qualified name without its module. The template links to it
    with a leading dot (``~.Class.member``), which Sphinx matches against the
    end of every documented name. That finds the page however the docs name
    the class (``pkg.Cls`` for a class defined in ``pkg.mod``, say), where a
    full name would miss.
    """
    if cls.__name__.startswith("_") or "<locals>" in cls.__qualname__:
        return None
    return cls.__qualname__


class ClassPage:
    """
    What a class page shows, for ``_templates/autosummary/class.rst``.

    A class's page gives a row and a page to the members it defines. Those
    it inherits from another class of the package are links to that class's
    pages instead, unless that class has no public name (a private mixin),
    when they count as the class's own.

    The template calls it as ``class_page.members(fullname, names)``, and so
    on. It is an object rather than functions so that Sphinx can pickle it
    with the configuration.
    """

    def members(self, fullname: str, names: list[str]) -> list[str]:
        """Return the names in *names* that get a row and a page."""
        cls = _import_class(fullname)
        if cls is None:
            return names
        return [
            name
            for name in names
            if _has_page(cls, name)
            and (_owner(cls, name) is cls or _link_name(_owner(cls, name)) is None)
        ]

    def inherited(self, fullname: str, names: list[str]) -> list[tuple[str, list[str]]]:
        """
        Return the members in *names* that link to a base class's pages.

        Each item is a base class's name and the names of its members, sorted, in
        the order of the MRO.
        """
        cls = _import_class(fullname)
        if cls is None:
            return []
        bases: dict[type, list[str]] = {}
        for name in names:
            owner = _owner(cls, name)
            if _has_page(cls, name):
                if owner is not cls:
                    bases.setdefault(owner, []).append(name)
            elif owner is not None and _defined_here(cls, name):
                # An undocumented override (a class constant set to another
                # value, say) links to the base class that documents it.
                for base in owner.__mro__[1:]:
                    if _package(base) == _package(cls) and _has_page(base, name):
                        bases.setdefault(_owner(base, name), []).append(name)
                        break
        found = []
        for base in cls.__mro__:
            path = _link_name(base) if base in bases else None
            if path:
                members = sorted(set(bases[base]), key=str.lower)
                found.append((path, [f"{path}.{name}" for name in members]))
        return found

    def has_bases(self, fullname: str) -> bool:
        """Whether the class has a base other than ``object``."""
        cls = _import_class(fullname)
        return cls is not None and cls.__bases__ != (object,)

    def is_enum(self, fullname: str) -> bool:
        """Whether the class is an enum, whose members its page lists inline."""
        cls = _import_class(fullname)
        return cls is not None and issubclass(cls, enum.Enum)


def _import_class(fullname: str) -> type | None:
    try:
        obj = import_by_name(fullname)[1]
    except ImportError:
        return None
    return obj if isinstance(obj, type) else None


def drop_member_sections(app, what, name, obj, options, lines) -> None:
    """Remove what the class page's tables list from a class docstring."""
    if what != "class":
        return
    if span := _section(lines, "Methods"):
        del lines[span[0] : span[1]]
    if span := _section(lines, "Attributes"):
        start, end = span
        kept = [
            line
            for attr, entry in _entries(lines[start + 2 : end])
            if not _has_page(obj, attr)
            for line in entry
        ]
        if any(line.strip() for line in kept):
            lines[start + 2 : end] = kept
        else:
            del lines[start:end]


def add_template_context(app, config) -> None:
    """Make ``class_page`` available to the autosummary templates."""
    config.autosummary_context.setdefault("class_page", ClassPage())


def setup(app):
    """Register the hooks, ahead of napoleon's (priority 500)."""
    app.connect("config-inited", add_template_context)
    app.connect("autodoc-process-docstring", drop_member_sections, priority=400)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
