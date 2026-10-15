from __future__ import annotations

import base64
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO, ImageGenProvider, error_response, resolve_aspect_ratio, save_b64_image, success_response)

try:
    from . import _agy
except ImportError:  # loaded as a top-level module (tests)
    import _agy  # type: ignore[no-redef]

_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
_RATIO_TEXT = {"landscape": "landscape 16:9", "portrait": "portrait 9:16", "square": "square 1:1"}
_MODEL = "agy-image-generator"


class AntigravityImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "antigravity"

    @property
    def display_name(self) -> str:
        return "Antigravity (agy subscription)"

    def is_available(self) -> bool:
        return _agy.agy_path() is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": _MODEL, "display": "agy image-generator subagent",
                 "speed": "~20-120s", "strengths": "Uses your Antigravity subscription; self-checks up to 3 attempts"}]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": self.display_name, "badge": "subscription", "tag": "Runs the agy CLI; no API key", "env_vars": []}

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": 3}

    def _stage_reference(self, ref: str, workdir: Path, idx: int) -> Optional[Path]:
        try:
            if ref.startswith(("http://", "https://")):
                dest = workdir / f"ref{idx}.png"
                with urllib.request.urlopen(ref, timeout=30) as r, open(dest, "wb") as f:  # noqa: S310
                    f.write(r.read(25 * 1024 * 1024))
                return dest
            if ref.startswith("data:") and ";base64," in ref:
                dest = workdir / f"ref{idx}.png"
                dest.write_bytes(base64.b64decode(ref.split(";base64,", 1)[1]))
                return dest
            src = Path(ref).expanduser()
            if src.is_file():
                dest = workdir / f"ref{idx}{src.suffix or '.png'}"
                shutil.copyfile(src, dest)
                return dest
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug("reference %s unusable: %s", ref, exc)
        return None

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, *, image_url: Optional[str] = None,
                 reference_image_urls: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        aspect = resolve_aspect_ratio(aspect_ratio)

        def fail(msg: str, kind: str = "provider_error") -> Dict[str, Any]:
            return error_response(error=msg, error_type=kind, provider="antigravity", model=_MODEL,
                                  prompt=prompt, aspect_ratio=aspect)

        if not isinstance(prompt, str) or not prompt.strip():
            return fail("Prompt is required and must be a non-empty string", "invalid_argument")
        if not self.is_available():
            return fail("agy binary not found; install Antigravity CLI", "auth_required")
        workdir = _agy.scratch_dir("agy-img-")
        refs = [r for r in ([image_url] if image_url else []) + list(reference_image_urls or []) if r][:3]
        staged = [p for i, r in enumerate(refs) if (p := self._stage_reference(r, workdir, i))]
        text = f"Generate an image ({_RATIO_TEXT[aspect]}): {prompt.strip()}\n"
        if staged:
            text += ("Use these images as " + ("the source to edit" if image_url else "style/composition references")
                     + ": " + ", ".join(str(p) for p in staged) + "\n")
        text += "Do not run any commands. Only generate the image and reply with its absolute file path."
        res = _agy.run_rotated(text, cwd=workdir, account=kwargs.get("account"),
                               timeout=float(kwargs.get("timeout") or 300))
        home = res["home"]
        try:
            files = _agy.brain_files(home, [res["conversation_id"], *res["subagent_ids"]], _SUFFIXES) if res["conversation_id"] else []
            if not files:
                return fail(res["error"] or f"agy produced no image. Reply: {res['response'][:300]}", "no_image")
            path = files[0]
            saved = save_b64_image(base64.b64encode(path.read_bytes()).decode("ascii"), prefix="antigravity",
                                   extension=path.suffix.lstrip(".").lower().replace("jpeg", "jpg"))
            return success_response(image=str(saved), model=_MODEL, prompt=prompt, aspect_ratio=aspect,
                                    provider="antigravity", modality="image" if staged else "text",
                                    extra={"agy_conversation_id": res["conversation_id"]})
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
