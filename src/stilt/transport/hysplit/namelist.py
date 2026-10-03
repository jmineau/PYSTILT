"""
Writer for HYSPLIT's ``SETUP.CFG`` namelist.

HYSPLIT v5 reads the old DEC/VMS ``$NAME ... $END`` namelist format. The
Fortran 90 ``&name ... /`` format that f90nml writes does not work.

HYSPLIT declares some parameters as INTEGER, such as ``RHB`` and ``RHT``.
Written as ``80.0``, such a value fails: the reader stops at the decimal
point and reads the rest as the next variable name. Whole-number floats are
therefore written without a decimal point (``80``).
"""

from pathlib import Path


class NameList:
    """
    Fortran namelist in the DEC/VMS format HYSPLIT reads.

    Parameters
    ----------
    group : str
        Namelist group name, such as ``SETUP``.
    """

    def __init__(self, group: str):
        self.group = group.upper()
        self._entries: list[tuple[str, str]] = []

    def add(self, key: str, value) -> None:
        """
        Append one key-value pair to the namelist.

        Parameters
        ----------
        key : str
            Parameter name (converted to uppercase).
        value : object
            Parameter value. Booleans, strings, lists of strings, and
            numbers are formatted for the Fortran reader.
        """
        self._entries.append((key.upper(), self._format(value)))

    def update(self, mapping: dict) -> None:
        """
        Append several key-value pairs.

        Parameters
        ----------
        mapping : dict
            Key-value pairs to add in iteration order.
        """
        for key, value in mapping.items():
            self.add(key, value)

    def write(self, path: str | Path) -> None:
        """
        Write the namelist, replacing any existing file.

        Parameters
        ----------
        path : str or Path
            File to write.
        """
        lines = [f"${self.group}"]
        for key, fmt in self._entries:
            lines.append(f"{key}={fmt},")
        lines.append("$END\n")
        Path(path).write_text("\n".join(lines))

    # ------------------------------------------------------------------

    @staticmethod
    def _format(value) -> str:
        """Format one value in Fortran namelist syntax."""
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
