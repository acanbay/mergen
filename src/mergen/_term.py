"""
Terminal colours for Mergen's messages.

Colours are written only where they are rendered: on a terminal and in
a Jupyter notebook. Output redirected to a file or a log, or captured,
stays plain text.
"""
from __future__ import annotations

import sys


def colour_on() -> bool:
    """Whether colour codes should be written right now."""
    isatty = getattr(sys.stdout, 'isatty', None)
    try:
        tty = bool(isatty()) if isatty is not None else False
    except (ValueError, OSError):           # e.g. a closed stream
        tty = False
    return tty or 'ipykernel' in sys.modules


class _Code:
    """An ANSI code that formats to itself when colour is on, else ''."""

    def __init__(self, code: str) -> None:
        self.code = code

    def __format__(self, spec: str) -> str:
        return self.code if colour_on() else ''

    def __str__(self) -> str:
        return self.__format__('')

    def __add__(self, other: str) -> str:
        return str(self) + other

    def __radd__(self, other: str) -> str:
        return other + str(self)


RED     = _Code("\033[0;31m")
GREEN   = _Code("\033[0;32m")
GREEN_B = _Code("\033[1;32m")
YELLOW  = _Code("\033[1;33m")
CYAN    = _Code("\033[0;36m")
RESET   = _Code("\033[0m")
