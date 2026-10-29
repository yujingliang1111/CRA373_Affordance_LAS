"""Reserve a dashboard TCP port using only Python's standard library.

This can run before Isaac Sim starts without importing NumPy or project modules
that depend on the simulation runtime.
"""
from __future__ import annotations

import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def reserve_http_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
    *,
    fallback_ports: int = 0,
) -> ThreadingHTTPServer:
    if not 0 <= port <= 65535 or fallback_ports < 0:
        raise ValueError("port must be 0..65535 and fallback_ports nonnegative")
    last_error = None
    for candidate in range(port, min(65535, port + fallback_ports) + 1):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
    raise last_error
