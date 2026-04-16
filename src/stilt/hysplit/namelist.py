"""Old-style DEC/VMS Fortran namelist writer for HYSPLIT SETUP.CFG.

HYSPLIT v5 expects the ``$NAME ... $END`` namelist format (DEC/VMS extension),
not the standard Fortran-90 ``&name ... /`` format that f90nml produces.

Additional constraint: HYSPLIT declares some parameters as INTEGER in its
Fortran source (e.g. RHB, RHT, FRHMAX, FRMR).  Writing a Python float like
``80.0`` for those fields causes a runtime error because the Fortran namelist
reader stops at the decimal point when scanning for an integer and then tries
to interpret the leftover ``.0rht`` as a new variable name.  To avoid this,
whole-number floats are written without a decimal point (``80`` not ``80.0``).
"""

from pathlib import Path


class NameList:
    """Accumulate key-value pairs and write a DEC/VMS-style Fortran namelist."""

    def __init__(self, group: str):
        self.group = group.upper()
        self._entries: list[tuple[str, str]] = []

    def add(self, key: str, value) -> None:
        """Append one key-value pair to the namelist.

        Parameters
        ----------
        key : str
            Parameter name (converted to uppercase).
        value : object
            Parameter value; formatted by :meth:`_format`.
        """
        self._entries.append((key.upper(), self._format(value)))

    def update(self, mapping: dict) -> None:
        """Append multiple key-value pairs from *mapping*.

        Parameters
        ----------
        mapping : dict
            Key-value pairs to add in iteration order.
        """
        for key, value in mapping.items():
            self.add(key, value)

    def write(self, path: str | Path) -> None:
        """Write the namelist to *path*, replacing any existing file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.unlink(missing_ok=True)
        lines = [f"${self.group}"]
        for key, fmt in self._entries:
            lines.append(f"{key}={fmt},")
        lines.append("$END\n")
        path.write_text("\n".join(lines))

    # ------------------------------------------------------------------

    @staticmethod
    def _format(value) -> str:
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, list):
            return ", ".join(f"'{v}'" for v in value)
        if isinstance(value, str):
            return repr(value) if value else ""
        # Write whole-number floats as integers to avoid Fortran INTEGER
        # field mismatch when HYSPLIT reads the namelist.
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        # Prevent scientific notation (e.g. 1e-05) - HYSPLIT's old-style
        # DEC/VMS namelist reader cannot parse it.
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)
