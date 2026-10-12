"""Hand Hermes' persona (SOUL.md + system prompt) to agy as its own workspace rules.

agy loads ``GEMINI.md`` from its workspace root as first-class rules, exactly as it does for a user
running ``agy`` in a project. The plugin writes the Hermes system prompt there, which Hermes already
builds with ``$HERMES_HOME/SOUL.md`` at its head, together with the tool-bridge contract. The per-turn
stdin message then carries only tools and conversation, so every call looks like an ordinary agy
session in a project that has rules.

If a system prompt arrives without the SOUL (a trimmed auxiliary call, say), the file on disk is
prepended so the persona never silently drops out. Set ``ANTIGRAVITY_WORKSPACE_RULES=0`` to fall back
to inlining everything in the prompt.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

RULES_FILENAME = "GEMINI.md"
_SOUL_PROBE_CHARS = 240


def rules_enabled() -> bool:
    return os.getenv("ANTIGRAVITY_WORKSPACE_RULES", "1").strip().lower() not in {"0", "false", "no", "off"}


def hermes_soul_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.getenv("HERMES_HOME") or Path.home() / ".hermes")
    return home / "SOUL.md"


def read_soul() -> str:
    try:
        return hermes_soul_path().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _soul_present(soul: str, system_text: str) -> bool:
    probe = " ".join(soul.split())[:_SOUL_PROBE_CHARS]
    return bool(probe) and probe in " ".join(system_text.split())


def build_rules(system_parts: Iterable[str], preamble: Iterable[str]) -> str:
    system_text = "\n\n".join(p for p in system_parts if p and p.strip())
    soul = read_soul()
    sections = ["# Hermes Agent workspace rules",
                "These rules come from Hermes Agent, which drives this session. Follow them for every reply."]
    if soul and not _soul_present(soul, system_text):
        sections.append(f"## Persona (SOUL.md)\n\n{soul}")
    if system_text:
        sections.append(f"## Hermes system prompt\n\n{system_text}")
    sections.append("## Tool contract\n\n" + "\n".join(f"- {line}" if not line.endswith(":") else line
                                                       for line in preamble))
    return "\n\n".join(sections).strip() + "\n"


def write_rules(workspace: str | Path, text: str) -> str:
    """Atomically write ``GEMINI.md``; return its sha256 so callers can respawn agy when it changes."""
    path = Path(workspace) / RULES_FILENAME
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return digest
    except OSError:
        pass
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return digest


def system_parts_of(messages: Iterable[dict[str, Any]], render) -> list[str]:
    parts = []
    for msg in messages:
        if isinstance(msg, dict) and str(msg.get("role") or "").strip().lower() == "system":
            if rendered := render(msg.get("content")):
                parts.append(rendered)
    return parts
