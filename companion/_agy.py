"""Run the official ``agy`` CLI headlessly and collect its stream-json events."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from . import _rotation
except ImportError:  # loaded as a top-level module (tests)
    import _rotation  # type: ignore[no-redef]

DEFAULT_MODEL = "gemini-3.8-flash-low"


def agy_path() -> Optional[str]:
    return shutil.which("agy") or (str(p) if (p := Path.home() / ".local/bin/agy").exists() else None)


def scratch_dir(prefix: str) -> Path:
    root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def run_agy(prompt: str, *, cwd: Path, model: str = DEFAULT_MODEL, home: Optional[Path] = None,
            timeout: float = 300.0) -> Dict[str, Any]:
    """One headless turn. Returns ``{ok, conversation_id, response, subagent_ids, error}``."""
    exe = agy_path()
    if not exe:
        return {"ok": False, "error": "agy binary not found on PATH", "conversation_id": "", "response": "",
                "subagent_ids": []}
    env = dict(os.environ)
    if home:
        env["HOME"] = str(home)
    cmd = [exe, "-p", prompt, "--model", model, "--output-format", "stream-json",
           "--print-timeout", f"{int(timeout)}s"]
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"agy timed out after {int(timeout)}s", "conversation_id": "",
                "response": "", "subagent_ids": []}
    conv, response, status, subs, err = "", "", "", [], ""
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("event")
        if kind == "init":
            conv = ev.get("conversation_id") or conv
        elif kind == "step_update":
            for sa in (ev["step_update"].get("subagent_info") or {}).get("subagents", []):
                if sa.get("conversation_id"):
                    subs.append(sa["conversation_id"])
        elif kind == "result":
            res = ev.get("result", {})
            conv = res.get("conversation_id") or conv
            response, status = res.get("response", ""), res.get("status", "")
    if status != "SUCCESS":
        err = (proc.stderr or proc.stdout or "").strip()[-400:] or f"agy status={status or 'none'}"
    return {"ok": status == "SUCCESS", "conversation_id": conv, "response": response,
            "subagent_ids": subs, "error": err}


def brain_files(home: Path, conversation_ids: List[str], suffixes: tuple) -> List[Path]:
    """Files agy saved as artifacts for these conversations, newest first."""
    found: List[Path] = []
    for cid in conversation_ids:
        d = home / ".gemini" / "antigravity-cli" / "brain" / cid
        if d.is_dir():
            found += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


_MAX_ATTEMPTS = 3


def run_rotated(prompt: str, *, cwd: Path, model: str = DEFAULT_MODEL, account: Optional[str] = None,
                timeout: float = 300.0) -> Dict[str, Any]:
    """``run_agy`` on the account the provider's rotation picks, failing over on quota / sign-in errors.

    Adds ``home`` (the HOME that served the call, where agy stored its artifacts) and ``account``.
    """
    mod = _rotation.provider_accounts()
    exclude: set = set()
    res: Dict[str, Any] = {"ok": False, "error": "no attempt", "conversation_id": "", "response": "",
                           "subagent_ids": []}
    for _ in range(_MAX_ATTEMPTS):
        acc = _rotation.pick(model, label=account, exclude=exclude)
        label = (acc or {}).get("label") or ""
        home = Path(acc["home_dir"]) if acc and acc.get("home_dir") else Path.home()
        if mod is not None and label:
            with mod.lease_account(label):
                res = run_agy(prompt, cwd=cwd, model=model, home=home if acc else None, timeout=timeout)
        else:
            res = run_agy(prompt, cwd=cwd, model=model, home=home if acc else None, timeout=timeout)
        res["home"], res["account"] = home, label
        if res["ok"]:
            if mod is not None and label:
                mod.update_last_used(label)
            return res
        failover = mod is not None and label and not account and (
            mod.is_quota_error(res["error"]) or mod.is_login_error(res["error"]))
        if not failover:
            return res
        mod.set_cooldown(label)
        exclude.add(label)
    return res
