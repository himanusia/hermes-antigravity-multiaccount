"""Debug utilities for dumping prompts sent to the Antigravity CLI."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import stat
import threading
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_counter = 0
_counter_lock = threading.Lock()


def _reset_counter(val: int = 0) -> None:
    """Reset the prompt debug dump counter (primarily for deterministic testing)."""
    global _counter
    with _counter_lock:
        _counter = val


def dump_prompt_debug(
    prompt: str,
    branch: str,
    model: str | None = None,
    messages: Sequence[dict[str, Any]] | None = None,
) -> Path | None:
    """Dump an outgoing prompt to ANTIGRAVITY_DEBUG_PROMPT_DIR if configured.

    If the environment variable is unset or empty, this function does nothing and incurs zero I/O.
    Any filesystem or permission errors are logged at DEBUG level and swallowed so user requests
    are never broken by diagnostic dumping.
    """
    debug_dir = os.environ.get("ANTIGRAVITY_DEBUG_PROMPT_DIR")
    if not debug_dir or not debug_dir.strip():
        return None

    try:
        dir_path = Path(debug_dir.strip())
        dir_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(dir_path, 0o700)
        except OSError:
            pass

        global _counter
        with _counter_lock:
            _counter += 1
            seq = _counter

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_{seq:04d}_{branch}.txt"
        file_path = dir_path / filename

        msg_list = list(messages or [])
        msg_count = len(msg_list)
        last_roles = [
            str(m.get("role") or "unknown")
            for m in msg_list[-6:]
            if isinstance(m, dict)
        ]
        roles_str = ", ".join(last_roles) if last_roles else "none"

        header_lines = [
            "=== PROMPT DEBUG DUMP ===",
            f"Model: {model or 'unknown'}",
            f"Branch: {branch}",
            f"Message Count: {msg_count}",
            f"Last Roles: {roles_str}",
            "=========================",
            "",
        ]
        content = "\n".join(header_lines) + prompt

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(file_path, flags, 0o600)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            os.chmod(file_path, 0o600)
        except OSError:
            pass

        return file_path
    except Exception as exc:
        logger.debug("Failed to dump prompt to debug directory: %s", exc)
        return None
