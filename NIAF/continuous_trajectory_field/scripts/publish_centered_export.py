"""Durably publish one immutable centered post-selection export unit."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import uuid
from pathlib import Path


REBASED_METADATA_FILES = (
    "export_summary.json",
    "sample_manifest_summary.json",
    "dtw_mpjpe_t2m_default_h2s_betas.json",
    "dtw_mpjpe_t2m_default_h2s_betas.csv",
    "dtw_mpjpe_t2m_pa_h2s_betas.json",
    "dtw_mpjpe_t2m_pa_h2s_betas.csv",
)


class CenteredExportPublicationError(RuntimeError):
    """Raised when an export unit cannot be published without mutation."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if error.errno not in unsupported:
                raise
    finally:
        os.close(descriptor)


def _rewrite_durably(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.rebasing.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CenteredExportPublicationError(
            "Atomic export publication requires renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise CenteredExportPublicationError(
                f"Refusing to replace existing export unit: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))


def rebase_and_publish_export_directory(build_dir: Path, final_dir: Path) -> None:
    """Rebase private metadata and atomically publish on the same filesystem."""

    raw_build = Path(build_dir).absolute()
    raw_final = Path(final_dir).absolute()
    if raw_build.is_symlink():
        raise CenteredExportPublicationError("Export build directory is a symlink")
    build_dir = raw_build.resolve()
    final_dir = raw_final.parent.resolve() / raw_final.name
    if (
        not build_dir.is_dir()
        or build_dir.is_symlink()
        or build_dir.parent != final_dir.parent
        or build_dir == final_dir
    ):
        raise CenteredExportPublicationError(
            "Export build/final paths are not same-directory real paths"
        )
    required = {"export_summary.json", "sample_manifest_summary.json"}
    for name in required:
        path = build_dir / name
        if not path.is_file() or path.is_symlink():
            raise CenteredExportPublicationError(
                f"Export build lacks required metadata: {name}"
            )
    for name in REBASED_METADATA_FILES:
        path = build_dir / name
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise CenteredExportPublicationError(
                f"Unsafe export metadata path: {path}"
            )
        payload = path.read_bytes().replace(
            os.fsencode(build_dir), os.fsencode(final_dir)
        )
        _rewrite_durably(path, payload)
    _fsync_directory(build_dir)
    _rename_directory_no_replace(build_dir, final_dir)
    _fsync_directory(final_dir.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build_dir", type=Path, required=True)
    parser.add_argument("--final_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rebase_and_publish_export_directory(args.build_dir, args.final_dir)


if __name__ == "__main__":
    main()
