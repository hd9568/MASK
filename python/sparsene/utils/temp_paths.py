"""Rebuildable SparseKG artifacts live outside the persistent run tree."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


DEFAULT_TEMP_ROOT = Path("/mnt/lustre-client/huoda/sparsekg_tmp")
TEMP_ROOT_ENV = "SPARSEKG_TMP_DIR"


def sparsekg_temp_root() -> Path:
    """Return the writable temporary root, refusing a missing default mount."""
    override = os.environ.get(TEMP_ROOT_ENV)
    root = Path(override).expanduser() if override else DEFAULT_TEMP_ROOT
    if not root.is_absolute():
        raise ValueError(f"{TEMP_ROOT_ENV} must be an absolute path: {root}")
    if not override and not Path("/mnt/lustre-client").is_mount():
        raise RuntimeError("/mnt/lustre-client is not mounted; refusing to write builds to the system disk")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError(f"temporary root is not writable: {root}")
    return root.resolve()


def sparsekg_build_dir(work_dir: Path, *, kind: str = "cmake") -> Path:
    """Give each run-owned work tree a stable, collision-resistant build path."""
    resolved = work_dir.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in resolved.name
    )[:48]
    return sparsekg_temp_root() / "builds" / kind / f"{safe_name}-{digest}"


def sparsekg_toolchain_env() -> dict[str, str]:
    """Route nvcc/CMake scratch files to the same temporary filesystem."""
    scratch = sparsekg_temp_root() / "toolchain_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("TMPDIR", "TMP", "TEMP"):
        env[key] = str(scratch)
    return env
