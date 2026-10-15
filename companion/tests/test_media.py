"""Media providers: account rotation with failover, image and STT paths (agy is mocked)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

import _agy  # noqa: E402
import image_provider  # noqa: E402
import stt_provider  # noqa: E402


def ok(conv="c1", response="done", subs=()):
    return {"ok": True, "conversation_id": conv, "response": response, "subagent_ids": list(subs), "error": ""}


def bad(error):
    return {"ok": False, "conversation_id": "", "response": "", "subagent_ids": [], "error": error}


class FakeAccounts:
    def __init__(self):
        self.cooled, self.used = [], []

    def lease_account(self, label):
        m = MagicMock()
        m.__enter__ = lambda s: None
        m.__exit__ = lambda s, *a: False
        return m

    is_quota_error = staticmethod(lambda e: "quota" in str(e).lower())
    is_login_error = staticmethod(lambda e: "log back in" in str(e).lower())

    def set_cooldown(self, label):
        self.cooled.append(label)

    def update_last_used(self, label):
        self.used.append(label)


class RunRotatedTest(unittest.TestCase):
    def run_with(self, accounts, results, **kw):
        fake = FakeAccounts()
        picks = iter(accounts)
        with patch.object(_agy._rotation, "provider_accounts", return_value=fake), \
             patch.object(_agy._rotation, "pick", side_effect=lambda *a, **k: next(picks)), \
             patch.object(_agy, "run_agy", side_effect=results) as run:
            out = _agy.run_rotated("p", cwd=Path("."), **kw)
        return out, fake, run

    def test_fails_over_on_quota_and_cools_down_the_account(self):
        a = {"label": "a", "home_dir": "/h/a"}
        b = {"label": "b", "home_dir": "/h/b"}
        out, fake, run = self.run_with([a, b], [bad("quota exceeded"), ok()])
        self.assertTrue(out["ok"])
        self.assertEqual(out["account"], "b")
        self.assertEqual(out["home"], Path("/h/b"))
        self.assertEqual(fake.cooled, ["a"])
        self.assertEqual(fake.used, ["b"])
        self.assertEqual(run.call_count, 2)

    def test_non_quota_error_does_not_fail_over(self):
        a = {"label": "a", "home_dir": "/h/a"}
        out, fake, run = self.run_with([a], [bad("boom")])
        self.assertFalse(out["ok"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(fake.cooled, [])

    def test_pinned_account_never_fails_over(self):
        a = {"label": "a", "home_dir": "/h/a"}
        out, fake, run = self.run_with([a], [bad("quota exceeded")], account="a")
        self.assertFalse(out["ok"])
        self.assertEqual(run.call_count, 1)

    def test_host_home_when_no_account(self):
        out, _, run = self.run_with([None], [ok()])
        self.assertEqual(out["home"], Path.home())
        self.assertIsNone(run.call_args.kwargs["home"])


class ImageProviderTest(unittest.TestCase):
    def test_returns_artifact_from_the_serving_accounts_brain_dir(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            brain = home / ".gemini/antigravity-cli/brain/conv1"
            brain.mkdir(parents=True)
            (brain / "pic.jpg").write_bytes(b"\xff\xd8jpegbytes")
            res = ok("conv1")
            res.update(home=home, account="a")
            with patch.object(_agy, "run_rotated", return_value=res), \
                 patch.object(image_provider, "save_b64_image", return_value=Path("/cache/x.jpg")) as save:
                out = image_provider.AntigravityImageGenProvider().generate("a cat", "portrait")
        self.assertTrue(out["success"], out)
        self.assertEqual(out["image"], "/cache/x.jpg")
        self.assertEqual(save.call_args.kwargs["extension"], "jpg")

    def test_no_file_is_an_error_not_a_fake_success(self):
        with tempfile.TemporaryDirectory() as td:
            res = ok("conv1", "sorry")
            res.update(home=Path(td), account="")
            with patch.object(_agy, "run_rotated", return_value=res):
                out = image_provider.AntigravityImageGenProvider().generate("a cat")
        self.assertFalse(out["success"])
        self.assertEqual(out["error_type"], "no_image")

    def test_empty_prompt_rejected(self):
        self.assertFalse(image_provider.AntigravityImageGenProvider().generate("  ")["success"])


class SttProviderTest(unittest.TestCase):
    def test_transcript_from_agy_response(self):
        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "a.ogg"
            audio.write_bytes(b"x")
            res = ok(response=" halo dunia \n")
            res.update(home=Path(td), account="")
            with patch.object(_agy, "run_rotated", return_value=res), \
                 patch.object(stt_provider.shutil, "which", return_value=None):
                out = stt_provider.AntigravityTranscriptionProvider().transcribe(str(audio), language="id")
        self.assertEqual(out, {"success": True, "transcript": "halo dunia", "provider": "antigravity"})

    def test_agy_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "a.wav"
            audio.write_bytes(b"x")
            res = bad("Authentication expired")
            res.update(home=Path(td), account="")
            with patch.object(_agy, "run_rotated", return_value=res), \
                 patch.object(stt_provider.shutil, "which", return_value=None):
                out = stt_provider.AntigravityTranscriptionProvider().transcribe(str(audio))
        self.assertFalse(out["success"])
        self.assertIn("Authentication expired", out["error"])


if __name__ == "__main__":
    unittest.main()
