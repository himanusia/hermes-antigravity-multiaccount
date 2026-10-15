"""Account selection for media calls: reuse the provider plugin's real rotation.

``antigravity-subscription-directsdk`` owns the registry, quota scoring, cooldowns and leases. It is a
sibling directory under ``plugins/``, so its ``accounts`` module is loaded as a submodule of a
synthetic package (its ``__init__`` is never executed). With no provider plugin or no registered
accounts every call falls through to the host HOME, exactly like rotation mode ``off``.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
PROVIDER_DIR_NAME = "antigravity-subscription-directsdk"
_PKG = "agy_directsdk_accounts"


def _provider_dir() -> Optional[Path]:
    candidates = [Path(__file__).resolve().parent.parent / PROVIDER_DIR_NAME,
                  Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes") / "plugins" / PROVIDER_DIR_NAME]
    return next((c for c in candidates if (c / "accounts.py").is_file()), None)


def provider_accounts() -> Any:
    """The provider plugin's ``accounts`` module, or ``None``."""
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None) or ""
        if f.endswith(f"{PROVIDER_DIR_NAME}/accounts.py") and hasattr(mod, "pick_account"):
            return mod
    root = _provider_dir()
    if root is None:
        return None
    try:
        if _PKG not in sys.modules:
            pkg = types.ModuleType(_PKG)
            pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
            sys.modules[_PKG] = pkg
        return importlib.import_module(f"{_PKG}.accounts")
    except Exception as exc:  # noqa: BLE001
        logger.warning("antigravity media: cannot load provider accounts: %s", exc)
        return None


def pick(model: str, *, label: Optional[str] = None, exclude: Optional[set] = None) -> Optional[dict]:
    """Account dict (``label``, ``home_dir``) or ``None`` for the host HOME."""
    mod = provider_accounts()
    if mod is None:
        return None
    label = (label or os.environ.get("ANTIGRAVITY_MEDIA_ACCOUNT") or "").strip()
    try:
        if label:
            for acc in mod.load_accounts().get("accounts", []):
                if label in (acc.get("label"), acc.get("id")):
                    return acc
            logger.warning("antigravity media: account %r not found; using rotation", label)
        return mod.pick_account(model=model, exclude_labels=exclude or set())
    except Exception as exc:  # noqa: BLE001
        logger.warning("antigravity media: pick failed open: %s", exc)
        return None
