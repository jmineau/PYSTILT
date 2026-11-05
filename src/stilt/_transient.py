"""
Whether an error is one a retry may get past.

``stilt``'s command group asks it of an error no command handled (3 if so,
4 if not), a worker of an error that ended a simulation (no failure is
recorded if so), and a met fetch of an error it could otherwise take for a
missing file.
"""

from __future__ import annotations

import errno
import socket
from collections.abc import Iterator

#: Errors from the cloud libraries that do not inherit from the built-in
#: ``ConnectionError`` and mean the network failed. Matched by name along an
#: exception's ``__mro__``, because these libraries are optional and not imported
#: here, so a base class stands for all its subclasses.
_NETWORK_ERRORS = frozenset(
    {
        # botocore's bases, for an endpoint, proxy, connect timeout or TLS failure
        # and for a read timeout or closed connection (the name is requests' too)
        "ConnectionError",
        "HTTPClientError",
        "IncompleteReadError",
        # aiohttp (s3fs, gcsfs, and fsspec's HTTP filesystem)
        "ClientConnectionError",
        "ClientPayloadError",
        # ftplib: a 4xx reply ("421 too many users"), transient by the FTP
        # standard; error_perm (5xx), error_reply and error_proto are not
        "error_temp",
    }
)

#: HTTP statuses of a service that is busy or briefly down. Any other 3xx, 4xx
#: or 5xx is an answer that a retry repeats: a bucket in another region, a bad
#: request, a 403 or a 404.
_BUSY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

#: Errnos of input/output trouble that goes away: a stale NFS handle, a busy or
#: interrupted call, a network that is down, and what s3fs turns a service's
#: "slow down" or internal error into. Python raises ``ConnectionError`` and
#: ``TimeoutError`` for the connection errnos, so they are not listed. ``EIO`` is
#: apart because it is also s3fs's errno for a code it does not know, so the
#: status in the chain decides first. Any other errno (EINVAL, ENOSPC, EROFS,
#: none at all) is a refusal or a full disk that a retry meets again.
_TRANSIENT_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "EREMOTEIO",
        "ESTALE",
        "EBUSY",
        "EAGAIN",
        "EINTR",
        "ENETDOWN",
        "ENETUNREACH",
        "EHOSTDOWN",
        "EHOSTUNREACH",
    )
    if hasattr(errno, name)
)


def _http_status(error: BaseException) -> int | None:
    """
    Return the HTTP status an error from a cloud or HTTP library carries, or None.

    Read from botocore's ``ClientError.response`` (a dict) and from the error's
    own ``status`` or ``code`` (aiohttp, gcsfs, ``urllib.error.HTTPError``).
    """
    response = getattr(error, "response", None)
    if isinstance(response, dict):  # botocore's ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    for attribute in ("status", "code"):
        status = getattr(error, attribute, None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


def _causes(error: BaseException) -> Iterator[BaseException]:
    """Yield an error and what it was raised from or while handling, outermost first."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = None if current.__suppress_context__ else current.__context__


def _says(error: BaseException) -> bool | None:
    """Return what this error says by its type, errno or status, or None."""
    if isinstance(error, ConnectionError | TimeoutError | socket.gaierror):
        return True
    if not _NETWORK_ERRORS.isdisjoint(c.__name__ for c in type(error).__mro__):
        return True
    if isinstance(error, OSError) and error.errno in _TRANSIENT_ERRNOS:
        return True
    status = _http_status(error)
    if status in _BUSY_STATUSES:
        return True
    if status is not None and status >= 300:
        return False
    return None


def is_transient(error: BaseException) -> bool:
    """
    Return whether a retry may get past *error*.

    True for a connection or I/O error, a busy service or an interrupted
    call; False for any other error, which a retry would repeat.

    The error and its ``__cause__`` or ``__context__`` chain are read,
    outermost first, and the first that says decides. An error says True by being
    a ``ConnectionError``, ``TimeoutError`` or name-resolution error, by a base
    class named in :data:`_NETWORK_ERRORS`, by an errno in
    :data:`_TRANSIENT_ERRNOS`, or by a busy service's HTTP status (408, 429,
    500, 502, 503, 504). It says False by any other 3xx, 4xx or 5xx status (a bucket
    in another region, a bad request, a 403, a 404). The chain comes first
    because s3fs turns a ``ClientError`` into a plain ``OSError`` and keeps it
    in ``__cause__``, and fsspec's HTTP filesystem raises ``FileNotFoundError``
    for every failure. If nothing in the chain says, ``EIO`` anywhere in it is True
    and anything else, ftplib's ``error_perm``, ``error_reply`` and
    ``error_proto`` included, is False.

    Parameters
    ----------
    error : BaseException
        The error to read.

    Returns
    -------
    bool
        True if a retry may get past it.
    """
    chain = list(_causes(error))
    for each in chain:
        verdict = _says(each)
        if verdict is not None:
            return verdict
    return any(isinstance(each, OSError) and each.errno == errno.EIO for each in chain)
