"""Runtime compatibility shims for third-party packages used by ATES."""

from __future__ import annotations

import os


def _relax_mmcv_version_guard() -> None:
    if os.environ.get("ATES_RELAX_MMCV_VERSION") != "1":
        return

    try:
        import mmcv  # type: ignore
    except Exception:
        return

    mmcv.__version__ = os.environ.get("ATES_MMCV_VERSION_OVERRIDE", "2.1.99")


_relax_mmcv_version_guard()

