"""Translate agy-native tool steps into Hermes tool calls.

Some Antigravity models (Claude in particular) ignore the ``<tool_call>`` text protocol and call agy's
built-in tools such as ``run_command``. The plugin never lets agy execute those on the host. Before
this module, the step was killed and the turn came back empty. Now the intent is re-issued as the
equivalent Hermes tool call, so Hermes' own tool, approval and sandbox layer runs it. Unmappable
native tools are still neutralized as before.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable

Mapper = Callable[[dict[str, Any]], dict[str, Any] | None]


def _pick(params: dict[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): v for k, v in params.items()}
    for n in names:
        v = lowered.get(n.lower())
        if v not in (None, ""):
            return v
    return None


def _terminal(p: dict[str, Any]) -> dict[str, Any] | None:
    cmd = _pick(p, "CommandLine", "command", "cmd")
    if not cmd:
        return None
    out: dict[str, Any] = {"command": str(cmd)}
    if cwd := _pick(p, "Cwd", "cwd", "WorkingDirectory"):
        out["workdir"] = str(cwd)
    return out


def _read_file(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "AbsolutePath", "path", "file_path", "FilePath")
    if not path:
        return None
    out: dict[str, Any] = {"path": str(path)}
    if (start := _pick(p, "StartLine", "offset")) is not None:
        try:
            out["offset"] = max(1, int(start))
        except (TypeError, ValueError):
            pass
    return out


def _write_file(p: dict[str, Any]) -> dict[str, Any] | None:
    path = _pick(p, "TargetFile", "path", "AbsolutePath")
    content = _pick(p, "CodeContent", "content", "Content")
    return {"path": str(path), "content": str(content)} if path and content is not None else None


def _search_content(p: dict[str, Any]) -> dict[str, Any] | None:
    pattern = _pick(p, "Query", "pattern", "SearchPattern")
    if not pattern:
        return None
    out: dict[str, Any] = {"pattern": str(pattern), "target": "content"}
    if path := _pick(p, "SearchPath", "path", "Directory"):
        out["path"] = str(path)
    return out


def _search_files(p: dict[str, Any]) -> dict[str, Any] | None:
    pattern = _pick(p, "Pattern", "pattern", "Query")
    if not pattern:
        return None
    out: dict[str, Any] = {"pattern": str(pattern), "target": "files"}
    if path := _pick(p, "SearchDirectory", "path", "DirectoryPath"):
        out["path"] = str(path)
    return out


def _web_search(p: dict[str, Any]) -> dict[str, Any] | None:
    q = _pick(p, "query", "Query")
    return {"query": str(q)} if q else None


def _web_extract(p: dict[str, Any]) -> dict[str, Any] | None:
    url = _pick(p, "Url", "url")
    return {"urls": [str(url)]} if url else None


# agy tool name -> (Hermes tool name, argument mapper)
NATIVE_TO_HERMES: dict[str, tuple[str, Mapper]] = {
    "run_command": ("terminal", _terminal),
    "view_file": ("read_file", _read_file),
    "write_to_file": ("write_file", _write_file),
    "grep_search": ("search_files", _search_content),
    "find_by_name": ("search_files", _search_files),
    "search_web": ("web_search", _web_search),
    "read_url_content": ("web_extract", _web_extract),
}


def tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names = set()
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def translate(step: dict[str, Any], available: set[str], index: int) -> SimpleNamespace | None:
    """Return a Hermes tool-call delta for an agy native tool step, or None if it cannot be mapped."""
    native = str(step.get("tool_name") or (step.get("tool_info") or {}).get("name") or "")
    mapping = NATIVE_TO_HERMES.get(native)
    if not mapping:
        return None
    hermes_name, mapper = mapping
    if hermes_name not in available:
        return None
    params = (step.get("tool_info") or {}).get("parameters")
    if not isinstance(params, dict):
        return None
    args = mapper(params)
    if args is None:
        return None
    ident = f"agy_{step.get('step_index', index)}_{native}"
    return SimpleNamespace(index=index, id=ident, type="function",
                           function=SimpleNamespace(name=hermes_name, arguments=json.dumps(args, ensure_ascii=False)))
