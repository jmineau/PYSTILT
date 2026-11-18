"""
Class and module pages for the API reference, in the style of pandas'.

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

A constant a class sets again in its body, which a base class of the package
declares, always gets a row and a page, so the page shows the value that
class gives it (``BB_205.model = '2b_205'``). The description is the base
class's ``#:`` comment, which autodoc inherits. Where there is none, the
class docstring's Attributes entry stays beside the row.

A variable the class declares with an annotation (a dataclass field) is data
even when its default is a function (``logger: Callable = print``): it is
documented as the field, with its ``#:`` comment, and never by the function's
signature and docstring. Described only in the class docstring, it gets no
row, like any other field.
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


def _is_method(owner: type, name: str) -> bool:
    """
    Whether *owner*'s member *name* is a method or property, rather than data.

    A variable the class declares with an annotation is data, even when its
    default is a function (a dataclass field such as ``logger: Callable =
    print``): the function is the value, not the member's own definition.
    """
    value = vars(owner).get(name)
    if not (callable(value) or isinstance(value, ROUTINES)):
        return False
    return not any(_annotated(klass, name) for klass in owner.__mro__)


def _annotated(klass: type, name: str) -> bool:
    """Whether *klass* annotates *name* in its own body."""
    try:
        return name in inspect.get_annotations(klass)
    except Exception:  # an annotation that cannot be evaluated
        return False


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
    if _is_method(owner, name):
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
    if _is_method(owner, name):
        return bool(getattr(vars(owner)[name], "__doc__", None))
    return (owner.__qualname__, name) in _attr_docs(owner.__module__)


def _is_routine(cls: type, name: str) -> bool:
    """Whether member *name* is a method or property, rather than data."""
    owner = _owner(cls, name)
    return owner is not None and _is_method(owner, name)


def _declared(klass: type, name: str) -> bool:
    """Whether *klass* declares *name*, with a value or an annotation alone."""
    try:
        return name in vars(klass) or name in inspect.get_annotations(klass)
    except Exception:  # an annotation that cannot be evaluated
        return name in vars(klass)


def _is_override(cls: type, name: str) -> bool:
    """
    Whether *cls* sets the constant *name* again, one a base of the package declares.

    That is data assigned in the class body (not a method or property) that
    another class of the package, earlier in the MRO than any other package,
    also declares.
    """
    if name not in vars(cls) or _is_method(cls, name):
        return False
    return any(
        _package(base) == _package(cls) and _declared(base, name)
        for base in cls.__mro__[1:]
    )


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
    owner = _owner(cls, name)
    if _has_own_doc(cls, name) or (owner and _is_override(owner, name)):
        return True
    # Undocumented: a row only for a method or property the class docstring
    # does not describe either.
    return _is_routine(cls, name) and name not in _described_attributes(cls)


def _page_describes(cls: type, name: str) -> bool:
    """
    Whether the page of member *name* has a description of its own.

    That is a docstring or ``#:`` comment on the member itself or, for a
    constant the class sets again, on the base class's declaration, which
    autodoc inherits.
    """
    if _has_own_doc(cls, name):
        return True
    owner = _owner(cls, name)
    return bool(owner and _is_override(owner, name)) and any(
        _has_own_doc(base, name) for base in owner.__mro__[1:] if _declared(base, name)
    )


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

    def fields(self, fullname: str, names: list[str]) -> list[str]:
        """
        Return the names in *names* that are fields with a function as default.

        autosummary lists such a member among the methods, because its value is
        a function. The class page lists it among the attributes, and
        ``_templates/autosummary/method.rst`` documents it as the attribute it is.
        """
        cls = _import_class(fullname)
        if cls is None:
            return []
        return [
            name
            for name in names
            if (owner := _owner(cls, name)) and not _is_method(owner, name)
        ]

    def is_field(self, fullname: str) -> bool:
        """Whether member *fullname* is a field with a function as default."""
        cls_name, _, name = fullname.rpartition(".")
        return name in self.fields(cls_name, [name])

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
            if not (_has_page(obj, attr) and _page_describes(obj, attr))
            for line in entry
        ]
        if any(line.strip() for line in kept):
            lines[start + 2 : end] = kept
        else:
            del lines[start:end]


class ModulePage:
    """
    What a module page shows besides what the module defines.

    A package often re-exports its public API from submodules and lists it in
    ``__all__``. autosummary leaves those names out of a module's page, which
    would leave the package's own page a bare list of submodules. This finds
    them so ``module.rst`` can list them, linking to the pages of the
    submodules that define them (no second page, which Sphinx would report as
    a duplicate). Like ``ClassPage``, it is an object so that Sphinx can
    pickle it with the configuration.
    """

    def exported(self, fullname: str) -> list[str]:
        """
        Return the public names *fullname* re-exports from its package.

        Each name is where the object is defined (relative to *fullname* when
        it is under it), for ``module.rst`` to link to. Only classes,
        exceptions and functions of the module's own package count, and only
        when ``__all__`` lists them.
        """
        try:
            module = import_by_name(fullname)[1]
        except ImportError:
            return []
        names = []
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name, None)
            if not (isinstance(obj, type) or inspect.isfunction(obj)):
                continue
            home = getattr(obj, "__module__", None)
            if home in (None, module.__name__):
                continue
            if home.split(".")[0] != module.__name__.split(".")[0]:
                continue
            if obj.__name__.startswith("_") or "<locals>" in obj.__qualname__:
                continue
            path = f"{home}.{obj.__qualname__}"
            # autosummary wants a name under the current module relative to it.
            names.append(path.removeprefix(f"{fullname}."))
        return names


def add_template_context(app, config) -> None:
    """Make ``class_page`` and ``module_page`` available to the autosummary templates."""
    config.autosummary_context.setdefault("class_page", ClassPage())
    config.autosummary_context.setdefault("module_page", ModulePage())


def setup(app):
    """Register the hooks, ahead of napoleon's (priority 500)."""
    # The hooks need autodoc's event and autosummary's config value, so load
    # both here: the extension then works wherever conf.py lists it.
    app.setup_extension("sphinx.ext.autodoc")
    app.setup_extension("sphinx.ext.autosummary")
    app.connect("config-inited", add_template_context)
    app.connect("autodoc-process-docstring", drop_member_sections, priority=400)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
