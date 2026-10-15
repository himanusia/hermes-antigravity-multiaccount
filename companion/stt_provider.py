from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from agent.transcription_provider import TranscriptionProvider

try:
    from . import _agy
except ImportError:  # loaded as a top-level module (tests)
    import _agy  # type: ignore[no-redef]


class AntigravityTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return "antigravity"

    @property
    def display_name(self) -> str:
        return "Antigravity (agy subscription)"

    def is_available(self) -> bool:
        return _agy.agy_path() is not None

    def list_models(self):
        return [{"id": _agy.DEFAULT_MODEL, "display": "Gemini 3.8 Flash (Low)"},
                {"id": "gemini-3.8-flash-medium", "display": "Gemini 3.8 Flash (Medium)"}]

    def transcribe(self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None,
                   **extra: Any) -> Dict[str, Any]:
        def fail(msg: str) -> Dict[str, Any]:
            return {"success": False, "transcript": "", "provider": "antigravity", "error": msg}

        if not self.is_available():
            return fail("agy binary not found")
        workdir = _agy.scratch_dir("agy-stt-")
        try:
            src = Path(file_path)
            dest = workdir / "audio.wav"
            if shutil.which("ffmpeg"):  # normalize any container/codec (ogg/opus, m4a, webm) to 16 kHz mono wav
                r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1",
                                    str(dest)], capture_output=True, text=True)
                if r.returncode != 0:
                    return fail(f"ffmpeg could not decode audio: {r.stderr.strip()[-200:]}")
            else:
                dest = workdir / ("audio" + (src.suffix or ".wav"))
                shutil.copyfile(src, dest)
            hint = f" The language is {language}." if language else ""
            vocab = f" Vocabulary hint: {extra['prompt']}." if extra.get("prompt") else ""
            text = (f"Transcribe the speech in {dest} verbatim.{hint}{vocab} Do not run commands. "
                    "Output only the transcript text, nothing else. If there is no speech, output nothing.")
            res = _agy.run_rotated(text, cwd=workdir, model=model or _agy.DEFAULT_MODEL,
                                   account=extra.get("account"), timeout=180)
            if not res["ok"]:
                return fail(res["error"])
            return {"success": True, "transcript": res["response"].strip(), "provider": "antigravity"}
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
