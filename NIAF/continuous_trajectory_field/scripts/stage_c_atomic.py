"""Linux/CIFS-safe atomic no-replace publication for Stage-C metadata."""

from __future__ import annotations

import ctypes
import errno
import os
import uuid
from pathlib import Path


RENAME_NOREPLACE = 1
AT_FDCWD = -100


class AtomicPublishError(RuntimeError):
    """An immutable Stage-C path could not be published exactly once."""


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOTSUP,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            }:
                raise
    finally:
        os.close(descriptor)


def rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing to replace ``destination``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise AtomicPublishError(
            "atomic no-replace publication requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            AT_FDCWD,
            os.fsencode(source),
            AT_FDCWD,
            os.fsencode(destination),
            RENAME_NOREPLACE,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise AtomicPublishError(
                f"immutable publication already exists: {destination}"
            )
        raise OSError(
            error_number, os.strerror(error_number), str(destination)
        )


def publish_bytes_no_replace(path: Path, encoded: bytes) -> None:
    """Durably publish one regular file without hard-link assumptions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        rename_no_replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
