"""
Exit codes, and the one that an unexpected error ends a command with.

``stilt`` and its callers share them. :func:`error_exit_code` reads an
error for whether a retry may get past it, which ``stilt``'s command group
asks of an error no command handled, and which a met fetch asks of one it
could otherwise take for a missing file.
"""

from __future__ import annotations

import errno
import socket
from collections.abc import Iterator

#: Exit codes. 2 is also Click's own for an unknown option or command, so a
#: driver never takes a typo for a run to retry. 3 says "run it again": the
#: run was stopped, or hit a connection or I/O error; what finished is on
#: disk. 4 says "do not": an unexpected error that a retry would repeat. 1 is
#: the code of a run whose simulations failed, and no other.
EXIT_COMPLETE, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED, EXIT_BUG = 0, 1, 2, 3, 4

#: Errors from the cloud libraries that do not inherit from ``OSError`` and
#: mean the network failed or the service is briefly unavailable. Matched by
#: class name along an exception's ``__mro__``, because botocore, aiohttp and
#: urllib3 are optional and not imported here.
_NETWORK_ERRORS = frozenset(
    {
        # botocore (also the errors s3fs and aiobotocore raise)
        "EndpointConnectionError",
        "ProxyConnectionError",
        "ConnectTimeoutError",
        "ReadTimeoutError",
        "ConnectionClosedError",
        "IncompleteReadError",
        # aiohttp (s3fs, gcsfs, and fsspec's HTTP filesystem)
        "ClientConnectionError",
        "ServerConnectionError",
        "ClientPayloadError",
        # urllib3
        "ProtocolError",
        # ftplib: a 4xx reply ("421 too many users"), transient by the FTP
        # standard; error_perm (5xx), error_reply and error_proto are not
        "error_temp",
    }
)

#: ``OSError`` subclasses that name a path, not the network: the file is not
#: there, or may not be used, and a retry meets the same answer.
_PATH_ERRORS = (
    FileNotFoundError,
    FileExistsError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
)

#: HTTP statuses of a service that is busy or briefly down. Any other 3xx, 4xx
#: or 5xx is an answer that a retry repeats: a bucket in another region, a bad
#: request, a 403 or a 404.
_BUSY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

#: Codes of botocore's ``ClientError`` for a service that asks the caller to slow
#: down or try again, whatever the status that carries them.
_BUSY_CODES = frozenset(
    {
        "Throttling",
        "ThrottlingException",
        "RequestLimitExceeded",
        "RequestThrottled",
        "SlowDown",
        "RequestTimeout",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
        "ServiceUnavailable",
        "InternalError",
        "OperationAborted",
    }
)

#: Errnos of an ``OSError`` that no type, status or cause explains, which are
#: input/output trouble that goes away: an I/O error, a stale NFS handle, a busy
#: or interrupted call, a connection that failed. Any other errno (EINVAL,
#: ENOSPC, EROFS, none at all) is a refusal or a full disk that a retry meets again.
_TRANSIENT_ERRNOS = frozenset(
    getattr(errno, name)
    for name in (
        "EIO",
        "EREMOTEIO",
        "ESTALE",
        "EBUSY",
        "EAGAIN",
        "EINTR",
        "ETIMEDOUT",
        "ECONNRESET",
        "ECONNREFUSED",
        "ECONNABORTED",
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

    Read from botocore's ``ClientError.response`` (a dict), from the
    ``status_code`` or ``status`` of the response a ``requests`` error holds,
    and from the error's own ``status``, ``status_code`` or ``code`` (aiohttp,
    gcsfs, ``urllib.error.HTTPError``).
    """
    response = getattr(error, "response", None)
    if isinstance(response, dict):  # botocore's ClientError
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return status if isinstance(status, int) else None
    holders = (response, error) if response is not None else (error,)
    for holder in holders:
        for attribute in ("status", "status_code", "code"):
            status = getattr(holder, attribute, None)
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


def _says(error: BaseException) -> int | None:
    """Return 3 or 4 when this one error says so by its status, code or type; else None."""
    response = getattr(error, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    status = _http_status(error)
    if code in _BUSY_CODES or status in _BUSY_STATUSES:
        return EXIT_INTERRUPTED
    if status is not None and status >= 300:
        return EXIT_BUG
    if isinstance(error, ConnectionError | TimeoutError | socket.gaierror):
        return EXIT_INTERRUPTED
    if not _NETWORK_ERRORS.isdisjoint(c.__name__ for c in type(error).__mro__):
        return EXIT_INTERRUPTED
    return None


def error_exit_code(error: Exception) -> int:
    """
    Return the exit code for an error the command did not expect.

    3 (:data:`EXIT_INTERRUPTED`, "run it again") when the error is a
    connection or I/O error that a retry may get past, 4
    (:data:`EXIT_BUG`, "do not") when it is any other.

    The error and its ``__cause__`` or ``__context__`` chain are read,
    outermost first, and the first that says decides. An error says 3 by a
    busy service's HTTP status (408, 429, 500, 502, 503, 504) or botocore
    code (:data:`_BUSY_CODES`), by being a ``ConnectionError``,
    ``TimeoutError`` or name-resolution error, or by a class named in
    :data:`_NETWORK_ERRORS` (which includes ftplib's ``error_temp``, a 4xx
    reply): the cloud libraries are optional, so the ``__mro__``'s class
    names are read, not the classes imported. It says 4 by any other 3xx, 4xx
    or 5xx status (a bucket in another region, a bad request, a 403, a 404).
    s3fs turns a ``ClientError`` into a plain ``OSError`` and keeps it in
    ``__cause__``, and fsspec's HTTP filesystem raises ``FileNotFoundError``
    for every failure, so the chain is read before a path error (a file that
    is not there or may not be used) is taken at its word. If nothing in the
    chain says, an ``OSError`` that is not about a path is 3 when its errno
    is input/output trouble that goes away (:data:`_TRANSIENT_ERRNOS`, such
    as s3fs's ``EIO`` for a code it does not know), and anything else,
    ftplib's ``error_perm``, ``error_reply`` and ``error_proto`` included,
    is 4.

    Parameters
    ----------
    error : Exception
        The error that ended the command.

    Returns
    -------
    int
        3 or 4.
    """
    chain = list(_causes(error))
    for each in chain:
        verdict = _says(each)
        if verdict is not None:
            return verdict
    transient = any(
        isinstance(each, OSError)
        and not isinstance(each, _PATH_ERRORS)
        and each.errno in _TRANSIENT_ERRNOS
        for each in chain
    )
    return EXIT_INTERRUPTED if transient else EXIT_BUG
