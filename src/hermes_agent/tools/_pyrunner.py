"""Subprocess entrypoint for the `run_python` tool. Not imported by the package.

Invoked as: python -I _pyrunner.py <script.py> <mem_mb> <cpu_seconds>

Applies what isolation the platform allows before handing control to user code:

* network: `socket` is neutered in-process, which stops every stdlib HTTP client
  (urllib, http.client, requests, httpx) because they all route through it.
* memory / CPU / file size / subprocess count: POSIX rlimits.
* Windows has no rlimit equivalent, so only the network block and the parent's
  wall-clock timeout apply there.

This is defence in depth against a confused model, NOT a security boundary
against hostile code. Run untrusted code in a container or VM.
"""

from __future__ import annotations

import contextlib
import runpy
import sys


def _block_network() -> None:
    import socket

    class BlockedSocket(socket.socket):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("Network access is disabled inside run_python.")

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise OSError("Network access is disabled inside run_python.")

    socket.socket = BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]
    if hasattr(socket, "create_server"):
        socket.create_server = _blocked  # type: ignore[assignment]


def _apply_rlimits(mem_mb: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:
        return  # Windows

    def _set(which: int, soft: int, hard: int | None = None) -> None:
        # A limit the platform refuses is skipped, not fatal.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(which, (soft, hard if hard is not None else soft))

    if mem_mb > 0:
        _set(resource.RLIMIT_AS, mem_mb * 1024 * 1024)
    if cpu_seconds > 0:
        _set(resource.RLIMIT_CPU, cpu_seconds)
    _set(resource.RLIMIT_FSIZE, 64 * 1024 * 1024)
    if hasattr(resource, "RLIMIT_NPROC"):
        _set(resource.RLIMIT_NPROC, 64)
    if hasattr(resource, "RLIMIT_CORE"):
        _set(resource.RLIMIT_CORE, 0)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _pyrunner.py <script.py> [mem_mb] [cpu_seconds]", file=sys.stderr)
        return 2

    script = sys.argv[1]
    mem_mb = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    cpu_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    _apply_rlimits(mem_mb, cpu_seconds)
    _block_network()

    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    except BaseException:
        import traceback

        # Hide the runner frames; the model only cares about its own traceback.
        exc_type, exc_value, tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, tb.tb_next if tb else None)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
